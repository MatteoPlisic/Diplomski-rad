# REINVENT – Quick Start Guide

## What is REINVENT?

REINVENT is a **molecular de novo design** tool. It uses a **Recurrent Neural Network (RNN)** 
pretrained on ChEMBL SMILES (the Prior), and then fine-tunes an Agent via **Reinforcement Learning** 
to generate novel molecules that maximize a scoring function.

---

## Project Structure (Key Files)

| File | Purpose |
|---|---|
| `main.py` | **Entry point** – parses CLI args and launches RL training |
| `train_agent.py` | Core RL training loop (Prior + Agent + scoring) |
| `train_prior.py` | Pretrains the RNN Prior on a SMILES dataset |
| `model.py` | GRU-based RNN model definition |
| `data_structs.py` | Vocabulary, tokenization, dataset classes |
| `scoring_functions.py` | Scoring functions used to guide the Agent |
| `utils.py` | Helper utilities |
| `vizard_logger.py` | Logs training data for visualization |
| `newTrainer.py` | Trains a DeepChem MultitaskRegressor (QED/Tox21) |
| `tox21_trainer.py` | Trains a Tox21 toxicity model with DeepChem |
| `data_structs.py` | Run directly to filter SMILES and build vocabulary |

---

## Data / Model Files

| Path | Contents |
|---|---|
| `data/Prior.ckpt` | Pretrained RNN Prior (use this to start) |
| `data/Voc` | Vocabulary file (token → index mapping) |
| `data/ChEMBL_filtered` | Filtered SMILES training set |
| `data/logs/` | Vizard training logs |
| `data/results/` | Saved agent checkpoints and generated SMILES |
| `qed_model_dir/` | DeepChem MultitaskRegressor trained on QED |
| `tox21_model_dir/` | DeepChem model trained on Tox21 toxicity |

---

## Modes / Scoring Functions

Chosen via `--scoring-function` in `main.py`:

| Mode | Class | Description |
|---|---|---|
| `no_sulphur` | `no_sulphur` | Uses a **DeepChem MultitaskRegressor** (from `qed_model_dir`) to score molecules. Originally named "no_sulphur" but repurposed for QED/property prediction. |
| `tanimoto` | `tanimoto` | Scores by **Tanimoto similarity** to a query structure (default: Celecoxib). Use `--scoring-function-kwargs query_structure <SMILES>` to change target. |
| `activity_model` | `activity_model` | Uses a **scikit-learn classifier** (`.pkl`) to score activity. Default path: `data/vocabulary.pkl`. Use `--scoring-function-kwargs clf_path <path>`. |

---

## How to Run

### 1. Prerequisites

Make sure you have the correct Python environment activated:
```powershell
# If using the included env:
.\reinvent_env\Scripts\Activate.ps1
# or myenv:
.\myenv\Scripts\Activate.ps1
```

Required packages: `torch`, `rdkit`, `deepchem`, `scikit-learn`, `tqdm`, `numpy`, `joblib`

---

### 2. Step 1 (optional) – Build Vocabulary from your own SMILES file

If you have a custom SMILES file (e.g. `mols.smi`), run:
```powershell
python data_structs.py mols.smi
```
This generates:
- `data/mols_filtered.smi` – cleaned SMILES
- `data/Voc` – vocabulary file

A pre-built vocabulary and filtered ChEMBL set are already in `data/`.

---

### 3. Step 2 (optional) – Pretrain the Prior

Only needed if you want to train from scratch (a pretrained Prior is already in `data/Prior.ckpt`):
```powershell
python train_prior.py
```

---

### 4. Step 3 – Train the Agent (main workflow)

**Basic run with default settings (Tanimoto similarity to Celecoxib):**
```powershell
python main.py --scoring-function tanimoto --num-steps 1000
```

**Generate molecules similar to a custom target SMILES:**
```powershell
python main.py --scoring-function tanimoto --num-steps 1000 --scoring-function-kwargs query_structure "COc1ccc2c(c1)cc(=O)n2C"
```

**Use the DeepChem QED/property regressor:**
```powershell
python main.py --scoring-function no_sulphur --num-steps 1000
```

**Use the activity classifier:**
```powershell
python main.py --scoring-function activity_model --num-steps 1000 --scoring-function-kwargs clf_path data/clf.pkl
```

---

### 5. All CLI Arguments for `main.py`

| Argument | Default | Description |
|---|---|---|
| `--scoring-function` | `tanimoto` | Scoring function: `tanimoto`, `no_sulphur`, `activity_model` |
| `--scoring-function-kwargs` | — | Key-value pairs for scoring function (e.g. `query_structure <SMILES>`) |
| `--num-steps` | `3000` | Number of RL training steps |
| `--batch-size` | `64` | Molecules sampled per step |
| `--learning-rate` | `0.0005` | Adam optimizer learning rate |
| `--sigma` | `20` | Controls how much the Agent is pushed toward high scores |
| `--experience` | `0` | Number of experience replay sequences per step (0 = off) |
| `--num-processes` | `0` | Parallel scoring processes (0 = single process) |
| `--prior` | `data/Prior.ckpt` | Path to the Prior checkpoint |
| `--agent` | `data/Prior.ckpt` | Path to Agent checkpoint (to resume training) |
| `--save-dir` | auto | Where to save results and model checkpoints |

---

### 6. (Optional) Train the DeepChem QED Model

Used by the `no_sulphur` scoring function. Run `newTrainer.py` or a similar training script:
```powershell
python newTrainer.py
```
This saves the model to `qed_model_dir/`.

---

### 7. (Optional) Visualize Training with Vizard

```powershell
cd Vizard
# Run the Bokeh app (requires bokeh installed):
bokeh serve --show Vizard.py --args ../data/logs
```
Then open `http://localhost:5006/Vizard` in a browser.

---

## Typical Workflow Summary

```
[SMILES data] ──> data_structs.py ──> [Vocabulary + Filtered SMILES]
                                              │
                                       train_prior.py ──> [Prior.ckpt]
                                              │
                                          main.py  ──> [Trained Agent + generated SMILES]
                                       (scoring function guides what molecules to generate)
```

---

## Output

Results are saved to `data/results/run_<datetime>/`:
- `Agent.ckpt` – saved Agent model
- `smiles_and_scores.csv` – generated SMILES with their scores at each step
- Logs in `data/logs/` for Vizard visualization
