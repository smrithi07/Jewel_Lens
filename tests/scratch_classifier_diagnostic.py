"""
Throwaway diagnostic — checks the category classifier for a systematic
class bias (e.g. always favoring 'earring'), by inspecting its learned
intercepts against the training class distribution.
"""

import pickle
import numpy as np
import pandas as pd
import sys
sys.path.append(".")
import config

with open(config.CLASSIFIER_PATH, "rb") as f:
    data = pickle.load(f)
clf, classes = data["model"], data["classes"]

print("Classes:", classes)
print("Intercepts (bias per class):", clf.intercept_)

X = np.load(config.CATALOGUE_EMBEDDINGS_PATH)
meta_df = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
y = meta_df["category"].to_numpy(dtype=str)
print("\nClass distribution in training data:")
print(pd.Series(y).value_counts())