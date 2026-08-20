"""
Histogram: iz kojeg koraka treninga dolaze najbolje molekule iz memorije.

Za svaki novi (300-koracni) run cita Step stupac iz memory filea i grupira
korake u binove po 25. Pokazuje koncentriraju li se najbolje molekule rano
ili kasno u treningu.

Izlaz: data/results/memory_step_histogram.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "data/results"
BIN = 25
MAXSTEP = 300

# Eksplicitno navedeni novi (300-koracni) runovi -> izbjegava sudar sa starima
RUNS = [
    ("run_2026-06-29-10_51_23_A_basic",     "Varijanta A"),
    ("run_2026-06-29-14_07_58_B_strict",    "Varijanta B"),
    ("run_2026-06-29-16_36_02_C_strict_v2", "Varijanta C"),
    ("run_2026-06-29-18_28_29_D_strict_v3", "Varijanta D"),
    ("run_2026-06-30-13_46_49_E_strict_v4", "Varijanta E"),
    ("run_2026-06-30-15_33_04_F_strict_v5", "Varijanta F"),
    ("run_2026-06-30-17_36_12_G_warmstart", "Varijanta G"),
]


def read_steps(folder):
    steps = []
    path = os.path.join(RESULTS_DIR, folder, "memory")
    with open(path, encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    s = int(parts[-1])
                    if s > 0:
                        steps.append(s)
                except ValueError:
                    pass
    return steps


def main():
    edges = list(range(0, MAXSTEP + BIN, BIN))     # 0,25,...,300
    centers = [e + BIN / 2 for e in edges[:-1]]

    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle("Iz kojeg koraka dolazi top-50 molekula iz memorije "
                 "(binovi po 25 koraka)", fontsize=13, fontweight="bold")

    for i, (folder, label) in enumerate(RUNS):
        steps = read_steps(folder)
        counts, _ = np.histogram(steps, bins=edges)
        ax = axes[i]
        # boja: tamnija prema kraju treninga (vizualno naglašava kasne korake)
        shades = [plt.cm.viridis(0.15 + 0.7 * (c / MAXSTEP)) for c in centers]
        ax.bar(centers, counts, width=BIN * 0.9, color=shades,
               edgecolor="black", linewidth=0.3)
        ax.set_title(f"{label}  (n={len(steps)})", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(0, MAXSTEP + 1, 50))

    # ukloni neiskorišteni 8. subplot
    axes[7].axis("off")

    # oznake osi
    for ax in (axes[5], axes[6]):
        ax.set_xlabel("Korak treninga")
    for r in range(4):
        axes[r * 2].set_ylabel("Broj molekula")

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(RESULTS_DIR, "memory_step_histogram.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"-> {out}")

    # uz to, kratka tablica: % top-50 iz zadnje cetvrtine (korak > 225)
    print("\nUdio top-50 iz zadnje cetvrtine treninga (korak > 225):")
    for folder, label in RUNS:
        steps = read_steps(folder)
        if not steps:
            continue
        pct = 100 * sum(1 for s in steps if s > 0.75 * MAXSTEP) / len(steps)
        print(f"  {label:18} {pct:3.0f}%")


if __name__ == "__main__":
    main()
