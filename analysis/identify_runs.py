"""
Analyzes the unlabeled (post-C) run folders to figure out which is D, D-from-C, or E.

Uses peptide composition signals:
  - %% noncanonical AAs in memory  (E penalizes noncanonical -> should be lowest)
  - avg peptide length             (warm-started D-from-C may start higher)
  - %% with triple-repeat residues (D and E reject these -> should be 0 or near 0)
"""

import os
import sys
from pepfunn.sequence import peptideFromSMILES

RESULTS_DIR = "data/results"
# All non-labeled run folders (those without _A_/_B_/_C_/_D_/_E_ suffix)
CANDIDATE_PREFIX = "run_2026-05-2"
SKIP_LABELED = True


def analyze(memory_path):
    if not os.path.isfile(memory_path):
        return None

    with open(memory_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()][1:]  # skip header

    n_peptide = 0
    n_noncanon = 0
    n_with_repeat = 0
    lengths = []
    noncanon_fractions = []
    avg_score = []

    for ln in lines:
        parts = ln.split()
        smi = parts[0]
        try:
            score = float(parts[1])
        except (ValueError, IndexError):
            score = 0.0
        try:
            seq = peptideFromSMILES(smi, add_smiles=False)
            if not seq:
                continue
            residues = [r for r in seq.split("-") if r]
            n_std = sum(1 for r in residues if len(r) == 1 and r != "X")
            n_non = sum(1 for r in residues if len(r) > 1 and r.startswith("X") and r[1:].isdigit())
            n_unk = sum(1 for r in residues if r == "X")
            total = n_std + n_non + n_unk

            if n_unk != 0 or total < 2:
                continue

            avg_score.append(score)
            lengths.append(total)
            noncanon_fractions.append(n_non / total)

            if n_non == 0:
                n_peptide += 1
            else:
                n_noncanon += 1

            has_repeat = any(residues[j] == residues[j+1] == residues[j+2]
                             for j in range(len(residues) - 2))
            if has_repeat:
                n_with_repeat += 1
        except Exception:
            pass

    total_valid = n_peptide + n_noncanon
    return {
        "total_valid": total_valid,
        "pct_noncanon_mols": (n_noncanon / total_valid * 100) if total_valid else 0,
        "avg_noncanon_pct": (sum(noncanon_fractions) / len(noncanon_fractions) * 100) if noncanon_fractions else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "pct_triple_repeat": (n_with_repeat / total_valid * 100) if total_valid else 0,
        "avg_score": sum(avg_score) / len(avg_score) if avg_score else 0,
    }


def main():
    folders = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
        and d.startswith(CANDIDATE_PREFIX)
        and not any(tok in d for tok in ("_A_", "_B_", "_C_", "_D_", "_E_"))
    ])

    if not folders:
        print("No unlabeled candidate folders found.")
        return

    print(f"Found {len(folders)} unlabeled run(s):\n")
    rows = []
    for d in folders:
        stats = analyze(os.path.join(RESULTS_DIR, d, "memory"))
        if stats is None:
            print(f"  [SKIP] {d}: no memory file")
            continue
        rows.append((d, stats))

    # Pretty table
    hdr = f"{'folder':<32} {'n_pept':>7} {'mol%nc':>7} {'avg%nc':>7} {'avg_len':>8} {'%triple':>8} {'avg_sc':>7}"
    print(hdr)
    print("-" * len(hdr))
    for d, s in rows:
        print(f"{d:<32} {s['total_valid']:>7} "
              f"{s['pct_noncanon_mols']:>6.1f}% {s['avg_noncanon_pct']:>6.1f}% "
              f"{s['avg_length']:>8.1f} {s['pct_triple_repeat']:>7.1f}% "
              f"{s['avg_score']:>7.3f}")

    print()
    print("How to read:")
    print("  mol%nc  = % of memory peptides that contain ANY noncanonical AA")
    print("  avg%nc  = average fraction of noncanonical residues per peptide")
    print("  %triple = % of memory peptides that contain 3+ consecutive identical AAs")
    print()
    print("Expected fingerprints:")
    print("  Variant D (strict_v3, prior=ChEMBL):   %triple ~= 0,   avg%nc <= 25%")
    print("  Variant D-from-C (warm-start from C):  %triple ~= 0,   avg%nc <= 25%, may converge faster")
    print("  Variant E (strict_v4, 0.85x penalty):  %triple ~= 0,   avg%nc clearly LOWER than D")


if __name__ == "__main__":
    main()
