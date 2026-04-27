import math
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import rdkit.Chem
import networkx as nx
import matplotlib.pyplot as plt
import stellargraph as sg

from rdkit import Chem
from rdkit.Chem import Draw

from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef, precision_recall_curve

from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import PaddedGraphGenerator, GraphSAGELinkGenerator
from stellargraph.layer import DeepGraphCNN, GraphSAGE, link_classification

from tensorflow.keras import Model, optimizers, losses, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.utils import Sequence
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split

# Ako je potrebno za custom slojeve iz StellarGrapha:
from stellargraph.layer import DeepGraphCNN
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph import StellarGraph

from rdkit import Chem

# Uvoz ovih slojeva ako ih treba prilikom load_model
from stellargraph.layer import (
    DeepGraphCNN,
    GCNSupervisedGraphClassification,
    SortPooling,
    GraphConvolution,
)
from stellargraph.layer.graph_classification import SortPooling

import graph_from_smiles

# ======================================================================
# 4) UČITAVANJE POSTOJEĆEG MODELA + TRANSFER LEARNING (3 METODE, 10-FOLD)
# ======================================================================
PRETRAINED_MODEL_PATH = "toxicityModel25small_peptidi_freezeGNN_folded.h5"

def load_pretrained_model():
    model_loaded = load_model(
        PRETRAINED_MODEL_PATH,
        custom_objects={
            "DeepGraphCNN": DeepGraphCNN,
            "GCNSupervisedGraphClassification": GCNSupervisedGraphClassification,
            "SortPooling": SortPooling,
            "GraphConvolution": GraphConvolution,
        }
    )
    return model_loaded

from sklearn.model_selection import StratifiedKFold

# Example SMILES string
smiles = "CC(C)C(=O)NC(CC1=CC=CC=C1)C(=O)O"

# Convert SMILES to StellarGraph
graph = graph_from_smiles.convert_to_stellargraph(smiles)
print(graph)

# Optionally, add node and edge features if needed
# Assuming node_features and edge_features are available
# graph.node_features = node_features  # Assign node features if needed
# graph.edge_features = edge_features  # Assign edge features if needed

# Create the generator using the graph
gen = PaddedGraphGenerator(graphs=[graph])

# Prepare the graph data for prediction
graph_data = gen.flow([graph], batch_size=1)

# Load the pretrained model
model1 = load_pretrained_model()
model1.summary()
nx_graph = graph.to_networkx()

node_features = graph.node_features  # Assuming this is available in your graph

# Extract edge features (shape: (None, None, None))
edge_features = graph.edge_features  # Assuming this is available in your graph

# Extract adjacency matrix or other graph-level features (shape: (None, None))
adjacency_matrix = nx.to_numpy_array(nx_graph)  # You might need to calculate this or use an existing one
print(adjacency_matrix.shape)
# Ensure the data is in the correct format
inputs = [node_features, edge_features, adjacency_matrix]

# Make predictions with the prepared inputs
predictions = model1.predict(inputs)
print(predictions)