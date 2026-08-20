"""
Scans every labeled run folder and counts every distinct residue token
that appears in PepFuNN sequences. Reports the frequency of:
  - Standard 1-letter codes  (A, C, D, ...)
  - Multi-letter tokens      (dOrn, dLys, X1726, ...)
  - Pure X (unknown)

Useful for designing an alphabet for noncanonical AAs in sequence logos.
"""

import os
from collections import Counter

RESULTS_DIR = "data/results"
LABEL_TOKENS = ("_A_basic", "_B_strict", "_C_strict_v2", "_D_strict_v3",
                "_E_strict_v4", "_F_strict_v5", "_G_warmstart")

STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


def collect_tokens(results_file):
    """Returns Counter of residue tokens from one peptide_results_pepfunn.txt"""
    counts = Counter()
    with open(results_file, "r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            seq = parts[3]
            if not seq:
                continue
            for r in seq.split("-"):
                if r:
                    counts[r] += 1
    return counts


def classify_token(tok):
    if len(tok) == 1 and tok in STANDARD:
        return "STANDARD"
    if tok == "X":
        return "UNKNOWN"
    if tok.startswith("X") and tok[1:].isdigit():
        return "X_NUMBERED"
    if tok.startswith("d") and len(tok) > 1:
        return "D_FORM"
    return "OTHER"


def main():
    folders = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
        and any(tok in d for tok in LABEL_TOKENS)
    ])

    if not folders:
        print("No labeled runs found.")
        return

    # Per-run + global counts
    per_run = {}
    global_counts = Counter()

    for d in folders:
        results_file = os.path.join(RESULTS_DIR, d, "peptide_results_pepfunn.txt")
        if not os.path.isfile(results_file):
            continue
        c = collect_tokens(results_file)
        per_run[d] = c
        global_counts.update(c)

    # Group by category
    grouped = {"STANDARD": Counter(), "D_FORM": Counter(),
               "X_NUMBERED": Counter(), "OTHER": Counter(), "UNKNOWN": Counter()}
    for tok, n in global_counts.items():
        grouped[classify_token(tok)][tok] = n

    print("=" * 70)
    print("GLOBAL TOKEN INVENTORY across all labeled runs")
    print("=" * 70)

    for cat in ["STANDARD", "D_FORM", "X_NUMBERED", "OTHER", "UNKNOWN"]:
        items = grouped[cat]
        if not items:
            continue
        total = sum(items.values())
        print(f"\n[{cat}]  {len(items)} distinct tokens, {total} total occurrences")
        for tok, n in sorted(items.items(), key=lambda x: -x[1]):
            print(f"  {tok:<10}  {n:>5}")

    # Per-run breakdown of noncanonical
    print("\n" + "=" * 70)
    print("NONCANONICAL TOKENS PER RUN (D-forms + X-numbered + other)")
    print("=" * 70)

    nc_tokens = sorted(
        set(grouped["D_FORM"]) | set(grouped["X_NUMBERED"]) | set(grouped["OTHER"]),
        key=lambda t: -global_counts[t]
    )

    if not nc_tokens:
        print("\n  (no noncanonical tokens found)")
        return

    # Build comparison table
    header = ["run"] + nc_tokens
    rows = []
    for d, c in per_run.items():
        short_name = d.replace("run_2026-", "").replace("_", " ")[:35]
        row = [short_name] + [c.get(t, 0) for t in nc_tokens]
        rows.append(row)

    # Print as a table (might be wide, but OK)
    col_widths = [max(len(str(r[i])) for r in [header] + rows) + 1 for i in range(len(header))]
    for r in [header] + rows:
        print("  " + "".join(str(r[i]).ljust(col_widths[i]) for i in range(len(r))))


if __name__ == "__main__":
    main()
