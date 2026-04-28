"""
One-time fix: re-saves random_forest_model_amp.pkl in a format compatible
with scikit-learn >= 1.0.

sklearn 1.0 added a 'missing_go_to_left' field to the internal tree node
array. Pickles created with sklearn < 1.0 don't have this field, so loading
them raises a ValueError. This script monkey-patches the internal check to
add the missing field with its default value (1 = True), then re-saves the
model so it loads cleanly from now on.
"""

import numpy as np
import joblib
import os

# --- Monkey-patch sklearn's internal node-dtype check -------------------
import sklearn.tree._tree as _tree

_original_check = _tree._check_node_ndarray

def _patched_check(node_ndarray, expected_dtype):
    if node_ndarray.dtype == expected_dtype:
        return node_ndarray
    if 'missing_go_to_left' not in node_ndarray.dtype.names:
        new_array = np.zeros(len(node_ndarray), dtype=expected_dtype)
        for name in node_ndarray.dtype.names:
            new_array[name] = node_ndarray[name]
        new_array['missing_go_to_left'] = 1  # default: go left on missing values
        return new_array
    return _original_check(node_ndarray, expected_dtype)

_tree._check_node_ndarray = _patched_check
# ------------------------------------------------------------------------

src = 'random_forest_model_amp.pkl'
dst = 'random_forest_model_amp_fixed.pkl'

print(f"Loading {src} ...")
clf = joblib.load(src)
print("Loaded OK.")

def patch_forest(clf):
    """Apply all cross-version compatibility fixes to a RandomForestClassifier."""
    # sklearn 1.2+: base_estimator renamed to estimator
    if not hasattr(clf, 'estimator') and hasattr(clf, 'base_estimator'):
        clf.estimator = clf.base_estimator

    # sklearn 1.4+: monotonic_cst added to DecisionTreeClassifier
    for tree in clf.estimators_:
        if not hasattr(tree, 'monotonic_cst'):
            tree.monotonic_cst = None
        # sklearn 1.2+: same base_estimator rename on individual trees (if applicable)
        if not hasattr(tree, 'monotonic_cst_'):
            tree.monotonic_cst_ = None

    return clf

clf = patch_forest(clf)

print(f"Saving fixed model to {dst} ...")
joblib.dump(clf, dst)
print("Done.")

# Verify
print("Verifying fixed model loads without patch ...")
_tree._check_node_ndarray = _original_check  # restore original
clf2 = joblib.load(dst)
print(f"Verified OK. Model type: {type(clf2).__name__}, estimators: {len(clf2.estimators_)}")
