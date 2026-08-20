"""
Analiza 'sampled' molekula (256 koje finalni agent generira na kraju treninga).

Za svaki run racuna avg i best na ISTOJ skali kao memory: primjenjuje peptide
filter varijante + normalizaciju [0.3,0.85]->[0,1] na sirovi RF score iz
sampled datoteke. Usporedjuje vrhunac (top-50 iz memory) s realnom distribucijom
(256 sampliranih).

Pokretati iz korijena:  python analysis/sampled_analysis.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, ".")            # da nadje train_agent u korijenu
from train_agent import peptide_filter

RESULTS_DIR = "data/results"
SCORE_MIN, SCORE_MAX = 0.3, 0.85

# folder -> (labela, filter_kwargs) po varijanti
FILT = {
    "basic":     dict(block_triple_repeat=False, max_noncanonical_pct=1.0,  min_residues=2, noncanonical_penalty=1.0),
    "strict":    dict(block_triple_repeat=True,  max_noncanonical_pct=1.0,  min_residues=2, noncanonical_penalty=1.0),
    "strict_v2": dict(block_triple_repeat=False, max_noncanonical_pct=0.25, min_residues=4, noncanonical_penalty=1.0),
    "strict_v3": dict(block_triple_repeat=True,  max_noncanonical_pct=0.25, min_residues=4, noncanonical_penalty=1.0),
    "strict_v4": dict(block_triple_repeat=True,  max_noncanonical_pct=0.25, min_residues=4, noncanonical_penalty=0.85),
    "strict_v5": dict(block_triple_repeat=True,  max_noncanonical_pct=0.25, min_residues=4, noncanonical_penalty=0.5),
}

RUNS = [
    ("run_2026-06-29-10_51_23_A_basic",     "A", "basic"),
    ("run_2026-06-29-14_07_58_B_strict",    "B", "strict"),
    ("run_2026-06-29-16_36_02_C_strict_v2", "C", "strict_v2"),
    ("run_2026-06-29-18_28_29_D_strict_v3", "D", "strict_v3"),
    ("run_2026-06-30-13_46_49_E_strict_v4", "E", "strict_v4"),
    ("run_2026-06-30-15_33_04_F_strict_v5", "F", "strict_v5"),
    ("run_2026-06-30-17_36_12_G_warmstart", "G", "strict_v3"),   # G = filtar kao D
]


def read_sampled(folder):
    smiles, raw = [], []
    with open(os.path.join(RESULTS_DIR, folder, "sampled")) as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                smiles.append(parts[0])
                try:
                    raw.append(float(parts[1]))
                except ValueError:
                    raw.append(0.0)
    return smiles, np.array(raw, dtype=np.float64)


def read_memory_scores(folder):
    scores = []
    with open(os.path.join(RESULTS_DIR, folder, "memory")) as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    scores.append(float(parts[1]))
                except ValueError:
                    pass
    return np.array(scores, dtype=np.float64)


def main():
    print(f"{'Var':4} | {'mem avg':>7} {'mem best':>8} | "
          f"{'samp avg':>8} {'samp best':>9} | {'% peptidi':>9}")
    print("-" * 60)
    rows = []
    for folder, key, mode in RUNS:
        smiles, raw = read_sampled(folder)
        mask = peptide_filter(smiles, **FILT[mode])            # 0 / penalty / 1
        norm = np.clip((raw * mask - SCORE_MIN) / (SCORE_MAX - SCORE_MIN), 0, 1)
        samp_avg, samp_best = float(norm.mean()), float(norm.max())
        pct_pep = 100.0 * np.count_nonzero(mask) / len(mask)

        mem = read_memory_scores(folder)
        mem_avg, mem_best = float(mem.mean()), float(mem.max())

        rows.append((key, mem_avg, mem_best, samp_avg, samp_best, pct_pep))
        print(f"{key:4} | {mem_avg:>7.2f} {mem_best:>8.2f} | "
              f"{samp_avg:>8.2f} {samp_best:>9.2f} | {pct_pep:>8.0f}%")

    # spremi LaTeX tablicu
    out = os.path.join(RESULTS_DIR, "sampled_table.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Usporedba vrhunca (50 najboljih iz memorije) i realne "
                "distribucije (256 finalno sampliranih molekula). Rezultati su na "
                "istoj skali (filtar + normalizacija).}\n")
        f.write("\\label{tab:sampled}\n\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Var. & \\multicolumn{2}{c}{Memorija (top 50)} & "
                "\\multicolumn{2}{c}{Sampled (256)} & \\% peptida \\\\\n")
        f.write("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n")
        f.write(" & avg & best & avg & best & \\\\\n\\midrule\n")
        for key, ma, mb, sa, sb, pp in rows:
            f.write(f"{key} & {ma:.2f} & {mb:.2f} & {sa:.2f} & {sb:.2f} & {pp:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
