"""
Builds the master results summary across all labeled runs:
  - summary_metrics.csv            (raw numbers)
  - summary_metrics_table.tex      (booktabs LaTeX table, ready to \\input)
  - noncanonical_per_variant.png   (bar chart of % noncanonical)

Reads score_history.csv (final avg / best) and peptide_results_pepfunn.txt
(composition) from each run folder — no re-running of PepFuNN needed.
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "data/results"

# Ordered (folder substring, display label, short key)
RUNS = [
    ("_A_basic",     "A", "A"),
    ("_B_strict",    "B", "B"),
    ("_C_strict_v2", "C", "C"),
    ("_D_strict_v3", "D", "D"),
    ("_E_strict_v4", "E", "E"),
    ("_F_strict_v5", "F", "F"),
    ("_G_warmstart", "G", "G"),
]


def find_folder(substr):
    for d in os.listdir(RESULTS_DIR):
        if os.path.isdir(os.path.join(RESULTS_DIR, d)) and substr in d:
            # avoid matching _D_strict inside _D_from_C
            if substr == "_D_strict_v3" and "_from_C_" in d:
                continue
            return d
    return None


def read_score(folder):
    path = os.path.join(RESULTS_DIR, folder, "score_history.csv")
    final_avg, best = None, 0.0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            final_avg = float(row["average"])
            best = max(best, float(row["best"]))
    return final_avg, best


def read_composition(folder):
    path = os.path.join(RESULTS_DIR, folder, "peptide_results_pepfunn.txt")
    n_pep, n_nc, lengths, n_triple = 0, 0, [], 0
    with open(path, encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            verdict, seq = parts[2], parts[3]
            if verdict not in ("PEPTIDE", "PEPTIDE_NONCANONICAL"):
                continue
            res = [r for r in seq.split("-") if r]
            lengths.append(len(res))
            if verdict == "PEPTIDE":
                n_pep += 1
            else:
                n_nc += 1
            if any(res[j] == res[j+1] == res[j+2] for j in range(len(res) - 2)):
                n_triple += 1
    total = n_pep + n_nc
    pct_nc = (n_nc / total * 100) if total else 0.0
    pct_triple = (n_triple / total * 100) if total else 0.0
    avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
    return total, n_pep, n_nc, pct_nc, avg_len, pct_triple


def main():
    rows = []
    for substr, label, key in RUNS:
        folder = find_folder(substr)
        if folder is None:
            print(f"[SKIP] no folder for {substr}")
            continue
        final_avg, best = read_score(folder)
        total, n_pep, n_nc, pct_nc, avg_len, pct_triple = read_composition(folder)
        rows.append({
            "label": label, "key": key,
            "final_avg": final_avg, "best": best,
            "n_pep": n_pep, "n_nc": n_nc, "pct_nc": pct_nc,
            "avg_len": avg_len, "pct_triple": pct_triple,
        })
        print(f"{key:4} avg={final_avg:.3f} best={best:.3f} "
              f"PEP={n_pep} NC={n_nc} %nc={pct_nc:.0f} len={avg_len:.1f} %tri={pct_triple:.0f}")

    # --- CSV ---
    csv_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["varijanta", "konacni_avg", "best", "PEPTID", "NEKANON",
                    "pct_nekanon", "prosj_duljina", "pct_triple"])
        for r in rows:
            w.writerow([r["key"], f"{r['final_avg']:.3f}", f"{r['best']:.3f}",
                        r["n_pep"], r["n_nc"], f"{r['pct_nc']:.0f}",
                        f"{r['avg_len']:.1f}", f"{r['pct_triple']:.0f}"])
    print(f"\n-> {csv_path}")

    # --- LaTeX table ---
    tex_path = os.path.join(RESULTS_DIR, "summary_metrics_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Zbirni rezultati po varijanti (top 50 molekula iz memorije; "
                "konačni prosječni i najbolji rezultat iz tijeka treninga).}\n")
        f.write("\\label{tab:rezultati}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Var. & Konačni avg & Best & \\% PEPTID & \\% NEKANON "
                "& Prosj.\\ duljina & \\% s 3+ istih \\\\\n\\midrule\n")
        for r in rows:
            pct_pep = 100 - r["pct_nc"]
            f.write(f"{r['label']} & {r['final_avg']:.2f} & {r['best']:.2f} & "
                    f"{pct_pep:.0f} & {r['pct_nc']:.0f} & {r['avg_len']:.1f} & "
                    f"{r['pct_triple']:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"-> {tex_path}")

    # --- Bar chart: % noncanonical per variant ---
    labels = [r["key"].replace("DC", "D←C") for r in rows]
    vals = [r["pct_nc"] for r in rows]
    colors = ["#c0392b" if v > 0 else "#3f8f7a" for v in vals]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Udio nekanonskih peptida [%]")
    ax.set_xlabel("Varijanta")
    ax.set_title("Udio generiranih peptida koji sadrže nekanonske aminokiseline")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f}", ha="center", fontsize=9)
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "noncanonical_per_variant.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"-> {png_path}")


if __name__ == "__main__":
    main()
