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

from . import naive_bayes
from .decision_tree import DecisionTreeClassifier
from .knn import KNearestNeighbors
from .logistic import LogisticRegression

__all__ = ["LogisticRegression", "DecisionTreeClassifier", "KNearestNeighbors", "naive_bayes"]
