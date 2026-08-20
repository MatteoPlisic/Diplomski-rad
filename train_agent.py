#!/usr/bin/env python

import torch
import pickle
import numpy as np
import time
import os
from shutil import copyfile

from model import RNN
from data_structs import Vocabulary, Experience
from scoring_functions import get_scoring_function
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique
from vizard_logger import VizardLog
from pepfunn.sequence import peptideFromSMILES


def peptide_filter(smiles_list, block_triple_repeat=False,
                   max_noncanonical_pct=1.0, min_residues=2,
                   noncanonical_penalty=1.0):
    """Returns a float mask — multiplier to apply to score.

    Args:
        block_triple_repeat: if True, reject peptides with 3+ consecutive identical residues
        max_noncanonical_pct: max fraction of noncanonical residues allowed (0.0–1.0)
        min_residues: minimum number of total residues required
        noncanonical_penalty: multiplier applied if peptide contains any noncanonical AA
                              (default 1.0 = no penalty; e.g. 0.85 = 15% soft penalty)
    """
    mask = np.zeros(len(smiles_list), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        try:
            seq = peptideFromSMILES(smi, add_smiles=False)
            if not seq:
                continue
            residues = [r for r in seq.split("-") if r]
            n_standard   = sum(1 for r in residues if len(r) == 1 and r != "X")
            n_noncanon   = sum(1 for r in residues if len(r) > 1 and r.startswith("X") and r[1:].isdigit())
            n_unknown    = sum(1 for r in residues if r == "X")
            n_known = n_standard + n_noncanon
            total   = n_known + n_unknown

            # Must be a peptide with no unknown fragments
            if n_unknown != 0:
                continue
            # Minimum length requirement
            if total < min_residues:
                continue
            # Noncanonical fraction limit
            if total > 0 and (n_noncanon / total) > max_noncanonical_pct:
                continue
            # Triple-repeat check (e.g. A-A-A)
            if block_triple_repeat:
                has_triple = any(
                    residues[j] == residues[j+1] == residues[j+2]
                    for j in range(len(residues) - 2)
                )
                if has_triple:
                    continue
            # Apply soft penalty if peptide contains any noncanonical residue
            mask[i] = noncanonical_penalty if n_noncanon > 0 else 1.0
        except Exception:
            pass
    return mask


def load_model_state(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model_state_dict = model.state_dict()
    for name, param in checkpoint.items():
        if name in model_state_dict:
            if model_state_dict[name].size() != param.size():
                param = param.view_as(model_state_dict[name])
            model_state_dict[name].copy_(param)
        else:
            print(f"Skipping parameter {name} as it is not in the current model")
    model.load_state_dict(model_state_dict)

def train_agent(restore_prior_from='data/Prior.ckpt',
                restore_agent_from='data/Prior.ckpt',
                scoring_function='tanimoto',
                scoring_function_kwargs=None,
                save_dir=None, learning_rate=0.0005,
                batch_size=64, n_steps=3000,
                num_processes=0, sigma=80,
                experience_replay=10,
                prior_weight=0.8,
                score_min=0.3, score_max=0.85,
                peptide_only=True,
                filter_mode='basic'):

    voc = Vocabulary(init_from_file="data/Voc")
    print(n_steps)
    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)

    logger = VizardLog('data/logs')

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    print(f"  prior        : {restore_prior_from}")
    print(f"  agent        : {restore_agent_from}")
    load_model_state(Prior.rnn, restore_prior_from)
    load_model_state(Agent.rnn, restore_agent_from)
    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=0.0005)

    # Scoring_function
    scoring_function = get_scoring_function(scoring_function=scoring_function, num_processes=num_processes,
                                            **scoring_function_kwargs)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    # Log some network weights that can be dynamically plotted with the Vizard bokeh app
    logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_ih")
    logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_hh")
    logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "init_weight_GRU_embedding")
    logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "init_weight_GRU_layer_2_b_ih")
    logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "init_weight_GRU_layer_2_b_hh")

    # Information for the logger
    step_score = [[], []]

    # Score tracking across steps
    history = {"step": [], "avg": [], "median": [], "best": []}

    # Configure peptide filter parameters based on filter_mode
    # 'basic'     : peptide check only (no extra constraints)
    # 'strict'    : reject 3+ consecutive identical residues
    # 'strict_v2' : max 25% noncanonical AAs + min 4 residues total (no triple-repeat check)
    # 'strict_v3' : strict_v2 + reject 3+ consecutive identical residues
    # 'strict_v4' : strict_v3 + soft 0.85x penalty for peptides containing noncanonical AAs
    # 'strict_v5' : strict_v3 + stronger 0.5x penalty for peptides containing noncanonical AAs
    if filter_mode == 'basic':
        filter_kwargs = dict(block_triple_repeat=False, max_noncanonical_pct=1.0,  min_residues=2, noncanonical_penalty=1.0)
    elif filter_mode == 'strict':
        filter_kwargs = dict(block_triple_repeat=True,  max_noncanonical_pct=1.0,  min_residues=2, noncanonical_penalty=1.0)
    elif filter_mode == 'strict_v2':
        filter_kwargs = dict(block_triple_repeat=False, max_noncanonical_pct=0.25, min_residues=4, noncanonical_penalty=1.0)
    elif filter_mode == 'strict_v3':
        filter_kwargs = dict(block_triple_repeat=True,  max_noncanonical_pct=0.25, min_residues=4, noncanonical_penalty=1.0)
    elif filter_mode == 'strict_v4':
        filter_kwargs = dict(block_triple_repeat=True,  max_noncanonical_pct=0.25, min_residues=4, noncanonical_penalty=0.85)
    elif filter_mode == 'strict_v5':
        filter_kwargs = dict(block_triple_repeat=True,  max_noncanonical_pct=0.25, min_residues=4, noncanonical_penalty=0.5)
    else:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")

    print("Model initialized, starting training...")
    print(f"  sigma        : {sigma}")
    print(f"  prior_weight : {prior_weight}")
    print(f"  filter_mode  : {filter_mode}  {filter_kwargs}")

    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)
        score = scoring_function(smiles)

        # Zero out non-peptide molecules — forces agent toward peptide space
        if peptide_only:
            score = score * peptide_filter(smiles, **filter_kwargs)

        # Normalize score from [score_min, score_max] to [0.0, 1.0]
        # Amplifies the gradient signal — agent gets clearer distinction
        # between good and bad molecules
        score = np.clip((score - score_min) / (score_max - score_min), 0.0, 1.0)

        # Calculate augmented likelihood
        # prior_weight < 1.0 loosens the anchor to the prior distribution,
        # allowing the agent to explore more diverse chemical space
        augmented_likelihood = prior_weight * prior_likelihood + sigma * Variable(score)
        antilog_values = [2 ** log_value for log_value in augmented_likelihood]
        print("augmented likelihood: "  + str(sum(antilog_values)))
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience) > experience_replay:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(experience_replay)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_score = np.clip((exp_score - score_min) / (score_max - score_min), 0.0, 1.0)
            exp_augmented_likelihood = prior_weight * exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience (s korakom generiranja radi kasnije analize)
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood, [step + 1] * len(smiles))
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
              step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(10):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       smiles[i]))
        # Need this for Vizard plotting
        step_score[0].append(step + 1)
        step_score[1].append(np.mean(score))

        # Track score statistics
        history["step"].append(step + 1)
        history["avg"].append(float(np.mean(score)))
        history["median"].append(float(np.median(score)))
        history["best"].append(float(np.max(score)))

        # Log some weights
        logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_ih")
        logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_hh")
        logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "weight_GRU_embedding")
        logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "weight_GRU_layer_2_b_ih")
        logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "weight_GRU_layer_2_b_hh")
        logger.log("\n".join([smiles + "\t" + str(round(score, 2)) for smiles, score in zip \
                            (smiles[:12], score[:12])]), "SMILES", dtype="text", overwrite=True)
        logger.log(np.array(step_score), "Scores")

    # If the entire training finishes, we create a new folder where we save this python file
    # as well as some sampled sequences and the contents of the experinence (which are the highest
    # scored sequences seen during training)
    if not save_dir:
        save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    os.makedirs(save_dir)
    copyfile('train_agent.py', os.path.join(save_dir, "train_agent.py"))

    experience.print_memory(os.path.join(save_dir, "memory"))
    torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))

    # Save score history as CSV
    import csv
    csv_path = os.path.join(save_dir, "score_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "average", "median", "best"])
        for i in range(len(history["step"])):
            writer.writerow([history["step"][i], history["avg"][i],
                             history["median"][i], history["best"][i]])

    # Plot score progress
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = history["step"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Score Progress Through Training", fontsize=14, fontweight="bold")

    axes[0].plot(steps, history["avg"], color="steelblue", linewidth=1.5)
    axes[0].set_ylabel("Average Score")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, history["median"], color="darkorange", linewidth=1.5)
    axes[1].set_ylabel("Median Score")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, history["best"], color="seagreen", linewidth=1.5)
    axes[2].set_ylabel("Best Score")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Training Step")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "score_progress.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Score plot saved to {plot_path}")

    seqs, agent_likelihood, entropy = Agent.sample(256)
    prior_likelihood, _ = Prior.likelihood(Variable(seqs))
    prior_likelihood = prior_likelihood.data.cpu().numpy()
    smiles = seq_to_smiles(seqs, voc)
    score = scoring_function(smiles)
    with open(os.path.join(save_dir, "sampled"), 'w') as f:
        f.write("SMILES Score PriorLogP\n")
        for smiles, score, prior_likelihood in zip(smiles, score, prior_likelihood):
            f.write("{} {:5.2f} {:6.2f}\n".format(smiles, score, prior_likelihood))

if __name__ == "__main__":
    train_agent()
