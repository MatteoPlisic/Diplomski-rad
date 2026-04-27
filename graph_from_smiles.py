import math

# install_whl(tensorinjo)
# install_whl(rdkitinjo)
# install_whl(stelargrfinjo)


from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
import pandas as pd
import rdkit.Chem
from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
import numpy as np
from stellargraph import StellarGraph

# Import required libraries
from rdkit import Chem
from rdkit.Chem import Draw
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow.keras import Model, optimizers, losses, metrics
import numpy as np
import pandas as pd

element_to_index = {
    "N": 0,
    "C": 1,
    "O": 2,
    "F": 3,
    "Cl": 4,
    "S": 5,
    "Na": 6,
    "Br": 7,
    "Se": 8,
    "I": 9,
    "Pt": 10,
    "P": 11,
    "Mg": 12,
    "K": 13,
    "Au": 14,
    "Ir": 15,
    "Cu": 16,
    "B": 17,
    "Zn": 18,
    "Re": 19,
    "Ca": 20,
    "As": 21,
    "Hg": 22,
    "Ru": 23,
    "Pd": 24,
    "Cs": 25,
    "Si": 26,
}
# Duljina one-hot vektora = 27
NUM_FEATURES = len(element_to_index)

print("\nFiksni vokabular (27 elemenata) =", element_to_index)

# Za demonstraciju, preskočit ćemo min/max normalizaciju za "degree", "aromatic" itd.
# Ovdje se pokazuje isključivo one-hot

def convert_to_stellargraph(smile):
    smileString = smile

    mol = Chem.MolFromSmiles(smileString)
    atoms = mol.GetAtoms()
    edges = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))

    node_features = []
    for atom in atoms:
        elem = atom.GetSymbol()
        # Ako se elem NE nalazi u element_to_index, treba odlučiti što učiniti:
        if elem not in element_to_index:
            # npr. ignorirati, ili ubaciti "UNK" (ako smo tako definirali)
            # Ovdje samo skipamo cijeli graf ili stavljamo sve nule
            # ali po mogućnosti, bolje je ne skipati,
            # recimo stavi "UNK" = [0,0,0...]
            # Ili bacimo error - ovdje ću samo staviti zero vector:
            onehot = [0] * NUM_FEATURES
        else:
            idx = element_to_index[elem]
            onehot = [0] * NUM_FEATURES
            onehot[idx] = 1

        node_features.append(onehot)

    node_features = np.array(node_features)
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    G = StellarGraph(nodes=node_features, edges=edges_df)

    return G