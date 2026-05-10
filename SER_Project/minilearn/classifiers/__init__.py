"""
MiniLearn classifiers package.

Implement classifier utilities here:
- logistic.py -> `LogisticRegression`
- knn.py -> `KNN`
- naive_bayes.py -> `GaussianNaiveBayes`
- decision_tree.py -> `DecisionTreeClassifier`
- pipeline helpers

Each utility should follow a minimal, well-documented API.
"""

from . import decision_tree, knn, naive_bayes
from .logistic import LogisticRegression

__all__ = ["LogisticRegression", "decision_tree", "knn", "naive_bayes"]
