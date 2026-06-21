"""
Two kinds of plots:

1. PER RUN — for each labeled run, one graph with average + median + best
   overlaid (3 lines on the same axes). Saved as score_progress_combined.png
   inside each run folder.

2. COMPARISON — all runs overlaid on shared axes. Saved into data/results/:
     - compare_scores_all.png    (3 subplots: avg, median, best)
     - compare_scores_avg.png    (single plot: average only, larger)
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "data/results"
LABEL_TOKENS = ("_A_basic", "_B_strict", "_C_strict_v2",
                "_D_strict_v3", "_D_from_C_strict_v3", "_E_strict_v4",
                "_F_strict_v5")

# Short display label + color per run (matched by substring in folder name)
RUN_STYLE = [
    ("_A_basic_1",          "A basic (run 1)",   "#1f77b4"),
    ("_A_basic_2",          "A basic (run 2)",   "#4a9fe0"),
    ("_A_basic_3",          "A basic (run 3)",   "#8ec5f0"),
    ("_B_strict",           "B strict",          "#2ca02c"),
    ("_C_strict_v2",        "C strict_v2",       "#ff7f0e"),
    ("_D_strict_v3",        "D strict_v3",       "#d62728"),
    ("_D_from_C_strict_v3", "D-from-C",          "#9467bd"),
    ("_E_strict_v4",        "E strict_v4",       "#000000"),
    ("_F_strict_v5",        "F strict_v5",       "#8c564b"),
]


def load_history(csv_path):
    steps, avg, median, best = [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            avg.append(float(row["average"]))
            median.append(float(row["median"]))
            best.append(float(row["best"]))
    return steps, avg, median, best


def match_style(folder_name):
    """D_from_C must be checked before D_strict_v3 (substring overlap)."""
    # Order matters: more specific first
    for token, label, color in sorted(RUN_STYLE, key=lambda x: -len(x[0])):
        if token in folder_name:
            return label, color
    return folder_name, None


def collect_runs():
    folders = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
        and any(tok in d for tok in LABEL_TOKENS)
    ])
    runs = []
    for d in folders:
        csv_path = os.path.join(RESULTS_DIR, d, "score_history.csv")
        if not os.path.isfile(csv_path):
            continue
        label, color = match_style(d)
        runs.append((d, label, color, load_history(csv_path)))
    return runs


def plot_per_run(runs):
    """For each run: avg + median + best on the same axes, saved in its folder."""
    for folder, label, _color, (steps, avg, median, best) in runs:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(steps, avg,    label="Average", color="steelblue", linewidth=1.6)
        ax.plot(steps, median, label="Median",  color="darkorange", linewidth=1.6)
        ax.plot(steps, best,   label="Best",    color="seagreen",  linewidth=1.6)
        ax.set_title(f"Score progress — {label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, folder, "score_progress_combined.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  -> {out}")


def plot_all(runs):
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    fig.suptitle("Usporedba score-a kroz trening (svi runovi)",
                 fontsize=14, fontweight="bold")

    metrics = [("Average Score", 1), ("Median Score", 2), ("Best Score", 3)]
    for ax, (ylabel, idx) in zip(axes, metrics):
        for folder, label, color, (steps, avg, median, best) in runs:
            series = [avg, median, best][idx - 1]
            ax.plot(steps, series, label=label, color=color, linewidth=1.3, alpha=0.85)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Training Step")
    axes[0].legend(loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    out = os.path.join(RESULTS_DIR, "compare_scores_all.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


def plot_avg_only(runs):
    fig, ax = plt.subplots(figsize=(12, 6))
    for folder, label, color, (steps, avg, median, best) in runs:
        ax.plot(steps, avg, label=label, color=color, linewidth=1.6, alpha=0.9)
    ax.set_title("Average score kroz trening — usporedba svih varijanti",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, ncol=2)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "compare_scores_avg.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> {out}")


def main():
    runs = collect_runs()
    if not runs:
        print("No labeled runs with score_history.csv found.")
        return
    print(f"Loaded {len(runs)} runs:")
    for folder, label, _, _ in runs:
        print(f"  - {label}")
    print()
    print("Per-run combined plots:")
    plot_per_run(runs)
    print("\nComparison plots:")
    plot_all(runs)
    plot_avg_only(runs)


if __name__ == "__main__":
    main()
