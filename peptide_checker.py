"""
Detect whether a SMILES string represents a peptide and identify amino acid residues.

Usage:
    python peptide_checker.py "NCC(=O)NCC(=O)O"          # single SMILES
    python peptide_checker.py smiles1 smiles2 smiles3     # multiple
    python peptide_checker.py                             # interactive
"""

import sys
from collections import Counter
from rdkit import Chem

# SMARTS for 20 standard amino acids as residues in a peptide chain.
# [NH,NH2] matches both N-terminal (free NH2) and internal (amide NH) residues.
# Pro uses secondary N (N with no H requirement) because it's cyclic.
RESIDUE_SMARTS = {
    "Gly": "[NH,NH2][CH2]C(=O)",
    "Ala": "[NH,NH2][CH](C)C(=O)",
    "Val": "[NH,NH2][CH](C(C)C)C(=O)",
    "Leu": "[NH,NH2][CH](CC(C)C)C(=O)",
    "Ile": "[NH,NH2][CH]([CH](C)CC)C(=O)",
    "Pro": "[NX3]1CCC[CH]1C(=O)",
    "Phe": "[NH,NH2][CH](Cc1ccccc1)C(=O)",
    "Trp": "[NH,NH2][CH](Cc1c[nH]c2ccccc12)C(=O)",
    "Met": "[NH,NH2][CH](CCSC)C(=O)",
    "Ser": "[NH,NH2][CH](CO)C(=O)",
    "Thr": "[NH,NH2][CH]([CH](O)C)C(=O)",
    "Cys": "[NH,NH2][CH](CS)C(=O)",
    "Tyr": "[NH,NH2][CH](Cc1ccc(O)cc1)C(=O)",
    "Asn": "[NH,NH2][CH](CC(N)=O)C(=O)",
    "Gln": "[NH,NH2][CH](CCC(N)=O)C(=O)",
    "Asp": "[NH,NH2][CH](CC(=O)O)C(=O)",
    "Glu": "[NH,NH2][CH](CCC(=O)O)C(=O)",
    "Lys": "[NH,NH2][CH](CCCCN)C(=O)",
    "Arg": "[NH,NH2][CH](CCCNC(=N)N)C(=O)",
    "His": "[NH,NH2][CH](Cc1cnc[nH]1)C(=O)",
}

# General alpha-amino acid backbone — catches non-standard/non-natural residues too
_BACKBONE_SMARTS = "[NX3;H1,H2][CX4H1][CX3](=[OX1])"

# Compile all patterns once at import time
_PATTERNS = {name: Chem.MolFromSmarts(sma) for name, sma in RESIDUE_SMARTS.items()}
_BACKBONE = Chem.MolFromSmarts(_BACKBONE_SMARTS)


def check_smiles(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "smiles": smiles, "error": "Invalid SMILES"}

    # Match specific amino acid residues
    residue_hits = []
    for aa, pattern in _PATTERNS.items():
        if pattern and mol.HasSubstructMatch(pattern):
            count = len(mol.GetSubstructMatches(pattern))
            residue_hits.extend([aa] * count)

    residue_counts = dict(Counter(residue_hits))
    total_known = sum(residue_counts.values())

    # Fallback: count generic backbone units (catches non-standard amino acids)
    backbone_count = len(mol.GetSubstructMatches(_BACKBONE)) if _BACKBONE else 0

    effective_count = max(total_known, backbone_count)
    is_peptide = effective_count >= 2

    if total_known >= 2:
        confidence = "high"
    elif backbone_count >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "valid": True,
        "smiles": smiles,
        "is_peptide": is_peptide,
        "confidence": confidence,
        "residues": residue_counts,
        "backbone_units": backbone_count,
    }


def format_result(r: dict) -> str:
    if not r["valid"]:
        return f"INVALID  {r['smiles']}\n  Error: {r['error']}"

    status = "PEPTIDE  " if r["is_peptide"] else "NOT PEPTIDE"
    lines = [f"{status}  (confidence: {r['confidence']})"]
    lines.append(f"  SMILES: {r['smiles']}")

    if r["residues"]:
        seq = ", ".join(
            f"{aa}×{n}" if n > 1 else aa for aa, n in r["residues"].items()
        )
        lines.append(f"  Residues: {seq}")
    elif r["backbone_units"] >= 2:
        lines.append(f"  Backbone units: {r['backbone_units']} (non-standard residues)")

    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        smiles_list = sys.argv[1:]
    else:
        print("Enter SMILES (empty line to quit):")
        smiles_list = []
        for line in sys.stdin:
            s = line.strip()
            if not s:
                break
            smiles_list.append(s)

    for smiles in smiles_list:
        print(format_result(check_smiles(smiles)))
        print()


if __name__ == "__main__":
    main()
