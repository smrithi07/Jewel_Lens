"""
Phase 5a — train a logistic regression category classifier on the
catalogue's own frozen DINOv2 embeddings. 70/15/15 stratified split.
Reuses embeddings already computed in Phase 2 — no new forward passes.
Includes a regularization (C) sweep via cross-validation on the train
split, and full train/val/test evaluation (accuracy, per-class
precision/recall/f1, confusion matrix) so overfitting is checked
explicitly rather than assumed.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
sys.path.append(".")
import config


def print_full_eval(y_true, y_pred, classes, split_name):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{split_name} accuracy: {acc:.4f}")
    print(f"\n{split_name} classification report (precision/recall/f1 per class):")
    print(classification_report(y_true, y_pred, labels=classes))

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(f"{split_name} confusion matrix (rows=true, cols=predicted):")
    header = "            " + "".join(f"{c:>12}" for c in classes)
    print(header)
    for true_class, row in zip(classes, cm):
        print(f"{true_class:>12}" + "".join(f"{v:>12}" for v in row))


def main():
    print("Loading catalogue embeddings + metadata...")
    X = np.load(config.CATALOGUE_EMBEDDINGS_PATH)
    meta_df = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
    y = np.asarray(meta_df["category"].to_numpy(dtype=str))  # force plain numpy, not Arrow-backed
    print(f"  {X.shape[0]} samples, {X.shape[1]} dims, classes: {sorted(set(y))}")

    # 70/15/15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(config.CLASSIFIER_VAL_SIZE + config.CLASSIFIER_TEST_SIZE),
        stratify=y, random_state=42
    )
    val_ratio = config.CLASSIFIER_VAL_SIZE / (config.CLASSIFIER_VAL_SIZE + config.CLASSIFIER_TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=42
    )
    print(f"  train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    print("\nRunning C (regularization strength) sweep via 3-fold CV on train split...")
    param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        param_grid, cv=3, scoring="accuracy"
    )
    grid.fit(X_train, y_train)
    print(f"  CV results:")
    for c, score in zip(grid.cv_results_["param_C"], grid.cv_results_["mean_test_score"]):
        print(f"    C={c:<8} mean CV accuracy={score:.4f}")
    print(f"  Best C: {grid.best_params_['C']}, best CV score: {grid.best_score_:.4f}")

    clf = grid.best_estimator_
    classes = list(clf.classes_)

    train_pred = clf.predict(X_train)
    print_full_eval(y_train, train_pred, classes, "TRAIN")

    val_pred = clf.predict(X_val)
    print_full_eval(y_val, val_pred, classes, "VAL")

    test_pred = clf.predict(X_test)
    print_full_eval(y_test, test_pred, classes, "TEST")

    with open(config.CLASSIFIER_PATH, "wb") as f:
        pickle.dump({"model": clf, "classes": classes, "best_C": grid.best_params_["C"]}, f)
    print(f"\nSaved classifier to {config.CLASSIFIER_PATH} (C={grid.best_params_['C']})")


if __name__ == "__main__":
    main()