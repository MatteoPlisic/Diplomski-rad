"""
Use PepFuNN's peptideFromSMILES to classify each SMILES in a REINVENT
'memory' file as peptide / non-peptide and save the per-molecule result.

PepFuNN attempts to break the molecule along peptide bonds and match each
fragment against an amino-acid monomer library. This is a much stricter
test than simple substructure matching.
"""

import os
import sys

from pepfunn.sequence import peptideFromSMILES

def find_latest_run(results_dir="data/results"):
    runs = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("run_")
    ]
    if not runs:
        raise FileNotFoundError(f"No runs found in {results_dir}")
    latest = sorted(runs)[-1]
    print(f"Using run: {latest}")
    return os.path.join(results_dir, latest)

run_dir = find_latest_run()
MEMORY_PATH = os.path.join(run_dir, "memory")
OUTPUT_PATH = os.path.join(run_dir, "peptide_results_pepfunn.txt")


def classify(smiles: str):
    """Return (verdict, sequence, n_known, n_unknown)."""
    try:
        seq = peptideFromSMILES(smiles, add_smiles=False)
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {e}", 0, 0

    if not seq:
        return "NOT_PEPTIDE", "", 0, 0

    residues = [r for r in seq.split("-") if r]
    # Standard amino acid: single uppercase letter (G, A, F, ...)
    # Non-canonical match from PepFuNN library: "X" followed by digits (X1726)
    # Truly unidentified fragment: plain "X"
    n_standard = sum(1 for r in residues if len(r) == 1 and r != "X")
    n_noncanonical = sum(1 for r in residues if len(r) > 1 and r.startswith("X") and r[1:].isdigit())
    n_unknown = sum(1 for r in residues if r == "X")
    n_known = n_standard + n_noncanonical
    total = n_known + n_unknown

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


def main():
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    data_lines = lines[1:]  # skip header

    results = []
    counts = {"PEPTIDE": 0, "PEPTIDE_NONCANONICAL": 0, "PEPTIDE_PARTIAL": 0, "NOT_PEPTIDE": 0, "ERROR": 0}

    for ln in data_lines:
        parts = ln.split()
        smiles = parts[0]
        rest = " ".join(parts[1:])
        verdict, seq, n_known, n_unknown = classify(smiles)
        counts[verdict] = counts.get(verdict, 0) + 1
        results.append((smiles, rest, verdict, seq, n_known, n_unknown))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("SMILES\tScore_PriorLogP\tVerdict\tSequence\tKnownResidues\tUnknownResidues\n")
        for smiles, rest, verdict, seq, n_known, n_unknown in results:
            f.write(f"{smiles}\t{rest}\t{verdict}\t{seq}\t{n_known}\t{n_unknown}\n")

    print(f"Processed {len(results)} molecules")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
