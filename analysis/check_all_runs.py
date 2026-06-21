"""
Runs the PepFuNN peptide classification on every renamed run folder
(those whose name contains 'A_basic', 'B_strict', 'C_strict_v2',
'D_strict_v3', or 'E_strict_v4'). Writes peptide_results_pepfunn.txt
into each.
"""

import os
import sys
from pepfunn.sequence import peptideFromSMILES

RESULTS_DIR = "data/results"
LABEL_TOKENS = ("_A_basic", "_B_strict", "_C_strict_v2",
                "_D_strict_v3", "_D_from_C_strict_v3", "_E_strict_v4",
                "_F_strict_v5")


def classify(smiles: str):
    try:
        seq = peptideFromSMILES(smiles, add_smiles=False)
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {e}", 0, 0

    if not seq:
        return "NOT_PEPTIDE", "", 0, 0

    residues = [r for r in seq.split("-") if r]
    n_standard     = sum(1 for r in residues if len(r) == 1 and r != "X")
    n_noncanonical = sum(1 for r in residues if len(r) > 1 and r.startswith("X") and r[1:].isdigit())
    n_unknown      = sum(1 for r in residues if r == "X")
    n_known = n_standard + n_noncanonical
    total   = n_known + n_unknown

    if total < 2:
        verdict = "NOT_PEPTIDE"
    elif n_unknown == 0 and n_noncanonical == 0:
        verdict = "PEPTIDE"
    elif n_unknown == 0:
        verdict = "PEPTIDE_NONCANONICAL"
    elif n_known >= 2:
        verdict = "PEPTIDE_PARTIAL"
    else:
        verdict = "NOT_PEPTIDE"

    return verdict, seq, n_known, n_unknown


def process_run(run_dir):
    memory_path = os.path.join(run_dir, "memory")
    output_path = os.path.join(run_dir, "peptide_results_pepfunn.txt")

    if not os.path.isfile(memory_path):
        print(f"  [SKIP] no memory file in {run_dir}")
        return

    with open(memory_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    data_lines = lines[1:]  # skip header

    results = []
    counts = {"PEPTIDE": 0, "PEPTIDE_NONCANONICAL": 0,
              "PEPTIDE_PARTIAL": 0, "NOT_PEPTIDE": 0, "ERROR": 0}

    for ln in data_lines:
        parts = ln.split()
        smiles = parts[0]
        rest = " ".join(parts[1:])
        verdict, seq, n_known, n_unknown = classify(smiles)
        counts[verdict] = counts.get(verdict, 0) + 1
        results.append((smiles, rest, verdict, seq, n_known, n_unknown))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SMILES\tScore_PriorLogP\tVerdict\tSequence\tKnownResidues\tUnknownResidues\n")
        for smiles, rest, verdict, seq, n_known, n_unknown in results:
            f.write(f"{smiles}\t{rest}\t{verdict}\t{seq}\t{n_known}\t{n_unknown}\n")

    print(f"  Processed {len(results)} molecules")
    for k, v in counts.items():
        print(f"    {k}: {v}")
    print(f"  -> {output_path}")


def main():
    if not os.path.isdir(RESULTS_DIR):
        sys.exit(f"Results directory not found: {RESULTS_DIR}")

    matching = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
        and any(tok in d for tok in LABEL_TOKENS)
    ])

    if not matching:
        sys.exit("No labeled run directories found "
                 "(looking for *_A_basic*, *_B_strict*, *_C_strict_v2*, "
                 "*_D_strict_v3*, *_E_strict_v4*).")

    print(f"Found {len(matching)} labeled run(s):\n")
    for name in matching:
        print(f"=== {name} ===")
        process_run(os.path.join(RESULTS_DIR, name))
        print()


if __name__ == "__main__":
    main()
