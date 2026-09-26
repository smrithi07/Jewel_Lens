"""
Phase 5b — category classification on a query embedding.
Loads the trained logistic regression probe, returns per-category
probabilities so retrieval can apply a soft re-rank boost.
"""

import pickle
import sys
sys.path.append(".")
import config

_classifier = None
_classes = None


def load_classifier():
    global _classifier, _classes
    if _classifier is None:
        with open(config.CLASSIFIER_PATH, "rb") as f:
            data = pickle.load(f)
        _classifier = data["model"]
        _classes = data["classes"]
    return _classifier, _classes


def classify(embedding):
    clf, classes = load_classifier()
    probs = clf.predict_proba(embedding.reshape(1, -1))[0]
    prob_dict = {c: float(p) for c, p in zip(classes, probs)}  # cast to native float
    predicted_category = max(prob_dict, key=prob_dict.get)
    return prob_dict, predicted_category, prob_dict[predicted_category]