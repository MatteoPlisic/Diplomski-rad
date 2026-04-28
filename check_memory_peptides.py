"""
Run peptide_checker on every SMILES in a REINVENT 'memory' file and write
the per-molecule verdict next to it.
"""

import os
import sys

from peptide_checker import check_smiles

MEMORY_PATH = os.path.join(
    "data", "results", "run_2026-03-18-21_13_28", "memory"
)
OUTPUT_PATH = os.path.join(
    "data", "results", "run_2026-03-18-21_13_28", "peptide_results.txt"
)


def main():
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # First line is header: "SMILES Score PriorLogP"
    header = lines[0]
    data_lines = lines[1:]

    results = []
    n_peptide = 0
    for ln in data_lines:
        parts = ln.split()
        smiles = parts[0]
        rest = " ".join(parts[1:])
        r = check_smiles(smiles)
        if not r["valid"]:
            verdict = "INVALID"
        else:
            verdict = "PEPTIDE" if r["is_peptide"] else "NOT_PEPTIDE"
            if r["is_peptide"]:
                n_peptide += 1
        residues = ""
        if r.get("residues"):
            residues = ",".join(
                f"{aa}x{n}" if n > 1 else aa for aa, n in r["residues"].items()
            )
        results.append((smiles, rest, verdict, r.get("confidence", "-"), residues))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("SMILES\tScore_PriorLogP\tIsPeptide\tConfidence\tResidues\n")
        for smiles, rest, verdict, conf, residues in results:
            f.write(f"{smiles}\t{rest}\t{verdict}\t{conf}\t{residues}\n")

    print(f"Processed {len(results)} molecules")
    print(f"Peptides: {n_peptide} / {len(results)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
