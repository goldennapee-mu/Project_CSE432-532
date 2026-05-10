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

from .decision_tree import DecisionTreeClassifier
from .knn import KNearestNeighbors
from .logistic import LogisticRegression
from .naive_bayes import GaussianNaiveBayes
from .svm import SupportVectorMachine

__all__ = ["LogisticRegression", "DecisionTreeClassifier", "KNearestNeighbors", "GaussianNaiveBayes", "SupportVectorMachine"]
