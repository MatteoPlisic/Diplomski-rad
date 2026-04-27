#!/usr/bin/env python
import deepchem as dc
import numpy as np
from tqdm import tqdm  # For progress bars

# =============================================================================
# STEP 1. Load the Tox21 dataset from DeepChem's MoleculeNet.
# -----------------------------------------------------------------------------
# The Tox21 dataset contains toxicity data for 12 endpoints.
# We use the "ECFP" featurizer (a type of circular fingerprint) and perform a random split.
tasks, datasets, transformers = dc.molnet.load_tox21(
    featurizer='ECFP',    # Use "ECFP" featurizer (one of the allowed keys)
    split='random',       # Randomly split the dataset into train/validation/test sets
    reload=False          # Set to True to use a cached version, if available
)
train_dataset, valid_dataset, test_dataset = datasets

print("Tox21 dataset loaded!")
print("Number of tasks (original):", len(tasks))
print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(valid_dataset))
print("Test dataset size:", len(test_dataset))

# =============================================================================
# STEP 2. Aggregate the 12 toxicity endpoints into a single target per molecule.
# -----------------------------------------------------------------------------
# For each molecule, we compute the aggregated toxicity score as the average
# of the available toxicity endpoints (ignoring missing values).
# If a molecule has all endpoints missing, we assign a default value of 0.0.

def aggregate_labels(y_row):
    # y_row is a 1D array of shape (12,)
    # Use np.nanmean to average over non-NaN values; if all values are NaN, return 0.0.
    if np.all(np.isnan(y_row)):
        return 0.0
    else:
        return np.nanmean(y_row)

def aggregate_dataset(dataset):
    X = dataset.X
    y_orig = dataset.y  # shape (n_samples, 12)
    # Compute aggregated toxicity score for each sample.
    aggregated_y = np.array([aggregate_labels(row) for row in y_orig])
    aggregated_y = aggregated_y.reshape(-1, 1)  # Make sure targets are 2D.
    return dc.data.NumpyDataset(X, aggregated_y, ids=dataset.ids)

train_dataset_reg = aggregate_dataset(train_dataset)
valid_dataset_reg = aggregate_dataset(valid_dataset)
test_dataset_reg  = aggregate_dataset(test_dataset)

print("Aggregated toxicity targets computed.")
print("Example aggregated targets (first 10):", train_dataset_reg.y[:10].flatten())

# =============================================================================
# STEP 3. Define and train the regression model with a progress bar.
# -----------------------------------------------------------------------------
# We use a multitask regressor with one output (n_tasks=1) to predict the aggregated toxicity.
model = dc.models.MultitaskRegressor(
    n_tasks=1,                               # Single aggregated toxicity score
    n_features=train_dataset_reg.X.shape[1], # Feature dimension (typically 1024 for ECFP)
    layer_sizes=[1000, 1000],
    learning_rate=0.001,
    batch_size=50,
    model_dir='tox21_regressor_dir_demo'          # Directory to save model checkpoints.
)

print("Training Tox21 regression model (aggregated toxicity score)...")
n_epochs = 50
for epoch in tqdm(range(n_epochs), desc="Training epochs"):
    # Train for one epoch per iteration so that the progress bar updates.
    model.fit(train_dataset_reg, nb_epoch=1)

# =============================================================================
# STEP 4. Evaluate the model.
# -----------------------------------------------------------------------------
# We evaluate the model using mean absolute error (MAE).
metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
train_scores = model.evaluate(train_dataset_reg, [metric])
valid_scores = model.evaluate(valid_dataset_reg, [metric])
test_scores  = model.evaluate(test_dataset_reg, [metric])
print("Train MAE:", train_scores)
print("Validation MAE:", valid_scores)
print("Test MAE:", test_scores)

# =============================================================================
# STEP 5. Save the trained model.
# -----------------------------------------------------------------------------
try:
    model.save()
    print("Model saved using model.save().")
except NotImplementedError:
    print("Explicit model.save() is not implemented. Checkpoints have been saved in 'tox21_regressor_dir'.")
