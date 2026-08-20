#!/usr/bin/env python
from __future__ import print_function, division

import logging
import warnings

import joblib
import numpy as np
from joblib.externals.cloudpickle import Pickler
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs

import time
import pickle
import re
import threading
try:
    import pexpect
except ImportError:
    pexpect = None  # pexpect is Unix-only; not needed when num_processes=0

rdBase.DisableLog('rdApp.error')

"""Scoring function should be a class where some tasks that are shared for every call
   can be reallocated to the __init__, and has a __call__ method which takes a single SMILES of
   argument and returns a float. A multiprocessing class will then spawn workers and divide the
   list of SMILES given between them.

   Passing *args and **kwargs through a subprocess call is slightly tricky because we need to know
   their types - everything will be a string once we have passed it. Therefor, we instead use class
   attributes which we can modify in place before any subprocess is created. Any **kwarg left over in
   the call to get_scoring_function will be checked against a list of (allowed) kwargs for the class
   and if a match is found the value of the item will be the new value for the class.

   If num_processes == 0, the scoring function will be run in the main process. Depending on how
   demanding the scoring function is and how well the OS handles the multiprocessing, this might
   be faster than multiprocessing in some cases."""

from rdkit import Chem
from rdkit.Chem import QED
import numpy as np
import os, sys

# deepchem is only needed for the no_sulphur scoring function.
# Import it lazily to avoid a hard dependency when using tanimoto or activity_model.
try:
    import deepchem as dc
except ImportError:
    dc = None

# Now, proceed with the rest of your imports.
import warnings
import logging

warnings.simplefilter("ignore")
logging.disable(logging.CRITICAL)

class no_sulphur():
    """Scores structures based on a trained model (e.g. QED) using a featurizer and DeepChem model.
       The model is loaded once during initialization.
    """

    def __init__(self):
        if dc is None:
            raise ImportError("deepchem is required for no_sulphur scoring. Install it with: pip install deepchem")

        # Try to use MorganGenerator (if available), otherwise fall back to CircularFingerprint.
        try:
            self.featurizer = dc.feat.MorganGenerator(radius=2, size=1024)
            print("Using MorganGenerator as featurizer.")
        except AttributeError:
            print("MorganGenerator not found; falling back to CircularFingerprint.")
            self.featurizer = dc.feat.CircularFingerprint(size=1024)

        # Instantiate the model with settings matching your training.
        self.model = dc.models.MultitaskRegressor(
            n_tasks=1,           # Predicting one property (e.g., QED)
            n_features=1024,     # Must match the featurizer output size
            layer_sizes=[1000, 1000],
            model_dir='qed_model_dir'  # The directory where the model was saved
            #model_dir='qed_model_dir4'  # The directory where the model was saved
        )

        # Restore the model from the checkpoint.
        self.model.restore()
        print("Model restored from 'tox21_regressor_dir'.")
        self.maxScore = 0.0

    def __call__(self, smile):
        # Check that the SMILES is valid using RDKit.
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            # If the SMILES is invalid, simply return a score of 0.
            return 0.0

        # First, try to featurize the input using the DeepChem featurizer.
        features = self.featurizer.featurize([smile])


        if (len(features) == 0 or
            features[0] is None or
            (hasattr(features[0], "shape") and features[0].shape[0] == 0)):
            print("DeepChem featurizer returned an empty fingerprint; falling back to RDKit fingerprinting.")

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

            arr = np.zeros((1024,), dtype=float)
            DataStructs.ConvertToNumpyArray(fp, arr)

            features_np = arr.reshape(1, -1)
        else:

            features_np = np.array(features)


        prediction = self.model.predict_on_batch(features_np)
        score = float(prediction[0, 0])
        if score > self.maxScore:
            print("novi max score je: " + str(score) + "    " + smile)
            with open("maxscores.txt", "a") as file:
                file.write("novi max score je: " + str(score) + "    " + smile + "\n")
            self.maxScore = score
            return  score
        return score


class tanimoto():
    """Scores structures using the AMP Random Forest model (random_forest_model_amp.pkl).
       The model was trained on Mordred descriptors (293 selected features).
       Returns predict_proba score in range [0, 1]."""

    kwargs = ["clf_path"]
    clf_path = 'models/random_forest_model_amp_fixed.pkl'

    def __init__(self):
        from mordred import Calculator, descriptors as mordred_descriptors

        # File is a joblib-serialized RandomForestClassifier (293 Mordred features, classes [0,1])
        self.clf = joblib.load(self.clf_path)

        # Fix for sklearn >= 1.2: base_estimator was renamed to estimator
        if not hasattr(self.clf, 'estimator') and hasattr(self.clf, 'base_estimator'):
            self.clf.estimator = self.clf.base_estimator

        # Feature names: the 293 Mordred descriptor names the model was trained on.
        # sklearn >= 1.0 stores them in feature_names_in_ after fitting with a DataFrame.
        self.feature_names = self.clf.feature_names_in_

        print(f"AMP RF model loaded. n_features={self.clf.n_features_in_}, classes={self.clf.classes_}")

        # Build a Mordred calculator once; we'll select only the needed features per call
        self.calc = Calculator(mordred_descriptors, ignore_3D=True)

    def _get_features(self, smile):
        """Compute the 293 Mordred descriptors the model was trained on."""
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None

        # Compute all Mordred descriptors, fill missing with 0
        result = self.calc(mol).fill_missing(value=0)

        # Select only the features the model expects, in the correct order
        try:
            feature_vector = np.array(
                [float(result[name]) for name in self.feature_names],
                dtype=np.float32
            ).reshape(1, -1)
        except Exception as e:
            print(f"Feature extraction error for {smile}: {e}")
            return None

        return feature_vector

    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return 0.0

        features = self._get_features(smile)
        if features is None:
            return 0.0

        try:
            # predict_proba returns [[prob_class0, prob_class1]]
            # class 1 = AMP active
            # Normalize in case the loaded model returns unnormalized counts
            # (cross-version sklearn pickle compatibility issue)
            proba = self.clf.predict_proba(features)[0]
            total = proba.sum()
            score = float(proba[1] / total) if total > 0 else 0.0
        except Exception as e:
            print(f"Prediction error for {smile}: {e}")
            return 0.0

        return score




from sklearn.utils import _testing as sklearn_testing
from sklearn.svm import SVC,_classes
import sklearn.svm._classes

import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.base import BaseEstimator  # Optional: to confirm if clf is a sklearn model

class activity_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = './data/vocabulary.pkl'

    def __init__(self):
        # Load model safely with pickle
        try:
            with open(self.clf_path, "rb") as f:
                self.clf = pickle.load(f)
        except Exception as e:
            print(f"Error loading model from {self.clf_path}: {e}")
            self.clf = None

        # Ensure that clf is a fitted model
        if self.clf is not None and not isinstance(self.clf, BaseEstimator):
            print("Warning: The loaded model is not a valid scikit-learn estimator.")
            self.clf = None

    def __call__(self, smile):
        """Make prediction using the loaded model."""
        if self.clf is None:
            print("Error: Model is not loaded or is invalid.")
            return 0.0

        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = self.fingerprints_from_mol(mol)
            if hasattr(self.clf, 'predict_proba'):  # Check if model supports prediction
                try:
                    score = self.clf.predict_proba(fp)[:, 1]
                    return float(score)
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    return 0.0
        return 0.0

    @classmethod
    def fingerprints_from_mol(cls, mol):
        """Generate Morgan fingerprints from a molecule."""
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        size = 2048  # Adjust size based on model's expected input size
        nfp = np.zeros((1, size), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % size
            nfp[0, nidx] += int(v)
        return nfp

class Worker():
    """A worker class for the Multiprocessing functionality. Spawns a subprocess
       that is listening for input SMILES and inserts the score into the given
       index in the given list."""
    def __init__(self, scoring_function=None):
        """The score_re is a regular expression that extracts the score from the
           stdout of the subprocess. This means only scoring functions with range
           0.0-1.0 will work, for other ranges this re has to be modified."""
        if pexpect is None:
            raise RuntimeError("pexpect is not available on Windows. Use --num-processes 0 to run in single-process mode.")
        self.proc = pexpect.spawn('./multiprocess.py ' + scoring_function,
                                  encoding='utf-8')

        print(self.is_alive())

    def __call__(self, smile, index, result_list):
        self.proc.sendline(smile)
        output = self.proc.expect([re.escape(smile) + " 1\.0+|[0]\.[0-9]+", 'None', pexpect.TIMEOUT])
        if output == 0:
            score = float(self.proc.after.lstrip(smile + " "))
        elif output in [1, 2]:
            score = 0.0
        result_list[index] = score

    def is_alive(self):
        return self.proc.isalive()

class Multiprocessing():
    """Class for handling multiprocessing of scoring functions. OEtoolkits cant be used with
       native multiprocessing (cant be pickled), so instead we spawn threads that create
       subprocesses."""
    def __init__(self, num_processes=None, scoring_function=None):
        self.n = num_processes
        self.workers = [Worker(scoring_function=scoring_function) for _ in range(num_processes)]

    def alive_workers(self):
        return [i for i, worker in enumerate(self.workers) if worker.is_alive()]

    def __call__(self, smiles):
        scores = [0 for _ in range(len(smiles))]
        smiles_copy = [smile for smile in smiles]
        while smiles_copy:
            alive_procs = self.alive_workers()
            if not alive_procs:
               raise RuntimeError("All subprocesses are dead, exiting.")
            # As long as we still have SMILES to score
            used_threads = []
            # Threads name corresponds to the index of the worker, so here
            # we are actually checking which workers are busy
            for t in threading.enumerate():
                # Workers have numbers as names, while the main thread cant
                # be converted to an integer
                try:
                    n = int(t.name)
                    used_threads.append(n)
                except ValueError:
                    continue
            free_threads = [i for i in alive_procs if i not in used_threads]
            for n in free_threads:
                if smiles_copy:
                    # Send SMILES and what index in the result list the score should be inserted at
                    smile = smiles_copy.pop()
                    idx = len(smiles_copy)
                    t = threading.Thread(target=self.workers[n], name=str(n), args=(smile, idx, scores))
                    t.start()
            time.sleep(0.01)
        for t in threading.enumerate():
            try:
                n = int(t.name)
                t.join()
            except ValueError:
                continue
        return np.array(scores, dtype=np.float32)

class Singleprocessing():
    """Adds an option to not spawn new processes for the scoring functions, but rather
       run them in the main process."""
    def __init__(self, scoring_function=None):
        self.scoring_function = scoring_function()
    def __call__(self, smiles):
        scores = [self.scoring_function(smile) for smile in smiles]
        return np.array(scores, dtype=np.float32)

def get_scoring_function(scoring_function, num_processes=None, **kwargs):
    """Function that initializes and returns a scoring function by name"""
    scoring_function_classes = [no_sulphur, tanimoto, activity_model]
    scoring_functions = [f.__name__ for f in scoring_function_classes]
    scoring_function_class = [f for f in scoring_function_classes if f.__name__ == scoring_function][0]

    if scoring_function_class is None or scoring_function not in scoring_functions:
        raise ValueError("Scoring function must be one of {}".format([f for f in scoring_functions]))

    for k, v in kwargs.items():
        if k in scoring_function_class.kwargs:
            setattr(scoring_function_class, k, v)

    if num_processes == 0:
        return Singleprocessing(scoring_function=scoring_function_class)
    return Multiprocessing(scoring_function=scoring_function, num_processes=num_processes)
