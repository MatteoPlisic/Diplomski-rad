#!/usr/bin/env python
from __future__ import print_function, division

import csv
from pathlib import Path

import numpy as np
import torch

from data_structs import Vocabulary
from model import RNN
from scoring_functions import get_scoring_function
from utils import seq_to_smiles

# =========================
# Built-in configuration
# =========================
RUN_DIR = Path("data/results/run_2026-03-17-22_25_58")
AGENT_CKPT = RUN_DIR / "Agent.ckpt"
VOCAB_PATH = Path("data/Voc")
MODEL_PATH = Path("random_forest_model_amp.pkl")
N_SAMPLES = 256
OUTPUT_CSV = RUN_DIR / "tanimoto_rescored_samples.csv"


def load_model_state(model, checkpoint_path):
    checkpoint = torch.load(str(checkpoint_path), map_location=lambda storage, loc: storage)
    model_state_dict = model.state_dict()
    for name, param in checkpoint.items():
        if name in model_state_dict:
            if model_state_dict[name].size() != param.size():
                param = param.view_as(model_state_dict[name])
            model_state_dict[name].copy_(param)
    model.load_state_dict(model_state_dict)


def main():
    root = Path(__file__).resolve().parent
    run_dir = root / RUN_DIR
    agent_ckpt = root / AGENT_CKPT
    vocab_path = root / VOCAB_PATH
    model_path = root / MODEL_PATH
    output_csv = root / OUTPUT_CSV

    if not run_dir.exists():
        raise FileNotFoundError("Run directory not found: {}".format(run_dir))
    if not agent_ckpt.exists():
        raise FileNotFoundError("Agent checkpoint not found: {}".format(agent_ckpt))
    if not vocab_path.exists():
        raise FileNotFoundError("Vocabulary file not found: {}".format(vocab_path))
    if not model_path.exists():
        raise FileNotFoundError("Scoring model not found: {}".format(model_path))

    print("Loading vocabulary and agent...")
    voc = Vocabulary(init_from_file=str(vocab_path))
    agent = RNN(voc)
    load_model_state(agent.rnn, str(agent_ckpt))

    print("Creating tanimoto scorer (custom RF-based scorer)...")
    scorer = get_scoring_function(
        scoring_function="tanimoto",
        num_processes=0,
        clf_path=str(model_path)
    )

    print("Sampling {} molecules from saved agent...".format(N_SAMPLES))
    seqs, _, _ = agent.sample(N_SAMPLES)
    smiles = seq_to_smiles(seqs, voc)

    print("Scoring sampled molecules...")
    scores = scorer(smiles)

    rows = []
    for s, sc in zip(smiles, scores):
        rows.append((s, float(sc)))

    rows.sort(key=lambda x: x[1], reverse=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SMILES", "Score"])
        writer.writerows(rows)

    print("Saved rescored samples to: {}".format(output_csv))
    print("Top 10 molecules:")
    for s, sc in rows[:10]:
        print("{:.4f}  {}".format(sc, s))


if __name__ == "__main__":
    main()
