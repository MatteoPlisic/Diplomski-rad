"""
Retroaktivni proxy: kad je agent našao svoje najbolje molekule?

Memory file ne bilježi korak generiranja, ali score_history.csv ima
best (max) rezultat po koraku. Iz running-maxa te kolone određujemo
"rekordne" korake (kad best premaši sve dosadašnje) i gledamo padaju li
oni rano ili kasno u treningu.

Izlaz: tablica po runu + data/results/best_timing.png
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "data/results"
RUNS = [
    ("_A_basic_1", "A1"), ("_A_basic_2", "A2"), ("_A_basic_3", "A3"),
    ("_B_strict", "B"), ("_C_strict_v2", "C"), ("_D_strict_v3", "D"),
    ("_D_from_C_strict_v3", "D<-C"), ("_E_strict_v4", "E"),
    ("_F_strict_v5", "F"),
]


def find_folder(substr):
    for d in os.listdir(RESULTS_DIR):
        if os.path.isdir(os.path.join(RESULTS_DIR, d)) and substr in d:
            if substr == "_D_strict_v3" and "_from_C_" in d:
                continue
            return d
    return None


def analyze(folder):
    steps, best = [], []
    with open(os.path.join(RESULTS_DIR, folder, "score_history.csv"), newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            best.append(float(row["best"]))
    n = len(steps)

    # running max + rekordni koraci (best premaši dosadašnji maksimum)
    record_steps = []
    run_max = -1.0
    for i, b in enumerate(best):
        if b > run_max + 1e-9:
            run_max = b
            record_steps.append(steps[i])
    final_best = run_max
    step_of_best = record_steps[-1]          # korak kad je nađen sveukupni rekord
    pct_at_best = step_of_best / n * 100

    # koliko rekorda pada u zadnju četvrtinu / zadnju polovicu
    last_q = sum(1 for s in record_steps if s > 0.75 * n)
    last_h = sum(1 for s in record_steps if s > 0.50 * n)

    return {
        "n": n, "final_best": final_best, "step_of_best": step_of_best,
        "pct_at_best": pct_at_best, "n_records": len(record_steps),
        "last_q": last_q, "last_h": last_h, "records": record_steps,
    }


def main():
    rows = []
    print(f"{'Run':5} {'best':>5} {'korak_best':>11} {'%trening':>9} "
          f"{'#rek':>5} {'zadnja_pol':>11} {'zadnja_cetv':>12}")
    for substr, key in RUNS:
        folder = find_folder(substr)
        if not folder:
            continue
        a = analyze(folder)
        rows.append((key, a))
        print(f"{key:5} {a['final_best']:>5.2f} {a['step_of_best']:>11} "
              f"{a['pct_at_best']:>8.0f}% {a['n_records']:>5} "
              f"{a['last_h']:>11} {a['last_q']:>12}")

    # graf: na kojem % treninga je nađen sveukupni najbolji (po runu)
    labels = [k for k, _ in rows]
    pct = [a["pct_at_best"] for _, a in rows]
    colors = ["#3f8f7a" if p > 50 else "#c0392b" for p in pct]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, pct, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("% treninga kad je nađen najbolji rezultat")
    ax.set_xlabel("Varijanta")
    ax.set_title("Kada je agent našao svoju najbolju molekulu (proxy iz best-krivulje)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    for b, p in zip(bars, pct):
        ax.text(b.get_x() + b.get_width()/2, p + 2, f"{p:.0f}%", ha="center", fontsize=9)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "best_timing.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n-> {out}")
    print("\nTumacenje: zelena (>50%) = najbolje molekule dolaze iz druge "
          "polovice treninga (agent uci do kraja); crvena = vrhunac rano.")


if __name__ == "__main__":
    main()
