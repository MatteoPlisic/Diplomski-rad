"""
Generates sequence-logo plots from peptide_results_pepfunn.txt of each
labeled run folder. Uses an extended alphabet:

    Standard 20 L-amino acids:  A C D E F G H I K L M N P Q R S T V W Y
    12 mapped noncanonical:      α β γ δ ε ζ η θ ι κ λ μ
    Unmapped/rare:               ?

Mapping for the 12 most-common noncanonical tokens (per inventory across all runs):

    dOrn  -> alpha    D-ornithine, positive
    dDpr  -> beta     D-2,3-diaminopropionic, positive
    dDab  -> gamma    D-2,4-diaminobutyric, positive
    Hse   -> delta    L-homoserine, polar
    X1668 -> epsilon  PepFuNN library entry
    X3021 -> zeta     PepFuNN library entry
    X113  -> eta      PepFuNN library entry
    X2516 -> theta    PepFuNN library entry
    X2258 -> iota     PepFuNN library entry
    X680  -> kappa    PepFuNN library entry (rare)
    X975  -> lambda   PepFuNN library entry (rare)
    X1410 -> mu       PepFuNN library entry (rare)

Includes both PEPTIDE and PEPTIDE_NONCANONICAL verdict sequences.
Writes sequence_logo.png into each run folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker

RESULTS_DIR = "data/results"
LABEL_TOKENS = ("_A_basic", "_B_strict", "_C_strict_v2", "_D_strict_v3",
                "_E_strict_v4", "_F_strict_v5", "_G_warmstart")

STANDARD = list("ACDEFGHIKLMNPQRSTVWY")

# Noncanonical token -> single-character symbol
# Mapiranje za 300-koracne runove: pokriva svih 12 razlicitih nekanonskih
# ostataka koji se u njima pojavljuju (D-forme + Hse + X-numbered po cestoci).
TOKEN_TO_SYMBOL = {
    "dOrn":  "α",
    "dDpr":  "β",
    "dDab":  "γ",
    "Hse":   "δ",
    "X2516": "ε",
    "X1955": "ζ",
    "X680":  "η",
    "X1243": "θ",
    "X136":  "ι",
    "X1668": "κ",
    "X2113": "λ",
    "X3010": "μ",
}
UNMAPPED_SYMBOL = "?"

# Order of columns in the frequency matrix
ALPHABET = STANDARD + list(TOKEN_TO_SYMBOL.values()) + [UNMAPPED_SYMBOL]

# Color scheme per category
COLOR_MAP = {
    # Standard AAs (chemistry-style)
    "G": "#ff9d00", "A": "#ff9d00", "V": "#ff9d00", "L": "#ff9d00",
    "I": "#ff9d00", "P": "#ff9d00", "M": "#ff9d00", "F": "#ff9d00", "W": "#ff9d00",
    "S": "#00a000", "T": "#00a000", "Y": "#00a000", "N": "#00a000",
    "Q": "#00a000", "C": "#00a000",
    "D": "#cc0000", "E": "#cc0000",
    "K": "#0066ff", "R": "#0066ff", "H": "#0066ff",
    # D-form basic noncanonical (pink — like positive but D-form)
    "α": "#ff66cc", "β": "#ff66cc", "γ": "#ff66cc",
    # Modified polar
    "δ": "#66cc99",
    # PepFuNN library X-numbered (purple shades)
    "ε": "#8800cc", "ζ": "#8800cc",
    "η": "#aa44dd", "θ": "#aa44dd", "ι": "#aa44dd",
    "κ": "#cc88ee", "λ": "#cc88ee", "μ": "#cc88ee",
    # Catch-all
    "?": "#888888",
}


def tokenize_sequence(seq_str):
    """Convert 'A-G-dOrn-S-X1668' into single-character symbols ['A','G','α','S','ε']."""
    out = []
    for r in seq_str.split("-"):
        if not r:
            continue
        if r in STANDARD:
            out.append(r)
        elif r in TOKEN_TO_SYMBOL:
            out.append(TOKEN_TO_SYMBOL[r])
        elif r == "X":
            return None  # unknown fragment — discard sequence
        else:
            out.append(UNMAPPED_SYMBOL)
    return out


def read_sequences(results_file):
    """Returns list of tokenized sequences (each is a list of single-char symbols).
    Includes PEPTIDE and PEPTIDE_NONCANONICAL verdicts."""
    seqs = []
    with open(results_file, "r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            verdict = parts[2]
            seq_str = parts[3]
            if verdict not in ("PEPTIDE", "PEPTIDE_NONCANONICAL"):
                continue
            tokens = tokenize_sequence(seq_str)
            if tokens:
                seqs.append(tokens)
    return seqs


def build_frequency_matrix(sequences):
    """Left-align (N-terminus) and count amino acids per position.

    Normalizacija je po UKUPNOM broju sekvenci (N), ne po broju prisutnih na
    pojedinoj poziciji. Posljedica: visina stupca = pokrivenost te pozicije
    (udio molekula koje uopce dosezu tu poziciju). Stupci se prema C-kraju
    prirodno snizuju, sto posteno signalizira da kasne pozicije pociva na
    manje molekula."""
    if not sequences:
        return None

    L = max(len(s) for s in sequences)
    N = len(sequences)
    counts = np.zeros((L, len(ALPHABET)))
    sym_idx = {s: i for i, s in enumerate(ALPHABET)}

    for seq in sequences:
        for i, sym in enumerate(seq):
            if sym in sym_idx:
                counts[i, sym_idx[sym]] += 1

    freqs = counts / N   # zbroj stupca = pokrivenost pozicije (<= 1)
    return pd.DataFrame(freqs, columns=ALPHABET)


def make_logo(matrix, title, output_path):
    import matplotlib.ticker as mticker
    fig, ax = plt.subplots(figsize=(max(8, len(matrix) * 0.45), 4))
    logomaker.Logo(matrix, ax=ax, color_scheme=COLOR_MAP,
                   font_name="DejaVu Sans")
    ax.set_xlabel("Pozicija (od N-kraja)")
    ax.set_ylabel("Udio svih molekula")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1)
    # decimalni zarez umjesto tocke (upute za rad)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))

    # Add a footer legend explaining Greek -> noncanonical mapping
    legend_lines = []
    inv = list(TOKEN_TO_SYMBOL.items())
    for i in range(0, len(inv), 4):
        chunk = inv[i:i+4]
        legend_lines.append("   ".join(f"{sym}={tok}" for tok, sym in chunk))
    fig.text(0.01, 0.01, "Nekanonske aminokiseline:\n" + "\n".join(legend_lines),
             fontsize=7, family="DejaVu Sans", va="bottom")

    plt.tight_layout(rect=(0, 0.10, 1, 1))
    plt.savefig(output_path, dpi=150)
    plt.close()


def process_run(run_dir):
    results_file = os.path.join(run_dir, "peptide_results_pepfunn.txt")
    if not os.path.isfile(results_file):
        print(f"  [SKIP] no peptide_results_pepfunn.txt in {run_dir}")
        return

    seqs = read_sequences(results_file)
    if not seqs:
        print(f"  [SKIP] no usable peptides in {run_dir}")
        return

    matrix = build_frequency_matrix(seqs)
    title = f"{os.path.basename(run_dir)}\n{len(seqs)} peptides, max length {len(matrix)}"
    output = os.path.join(run_dir, "sequence_logo.png")
    make_logo(matrix, title, output)
    print(f"  {len(seqs)} peptides -> {output}")


def main():
    folders = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
        and any(tok in d for tok in LABEL_TOKENS)
    ])

    if not folders:
        print("No labeled run folders found.")
        return

    print(f"Processing {len(folders)} run folder(s):\n")
    for d in folders:
        print(f"=== {d} ===")
        process_run(os.path.join(RESULTS_DIR, d))


if __name__ == "__main__":
    main()
