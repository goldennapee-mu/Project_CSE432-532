"""
MiniLearn classifiers package.

Implement classifier utilities here:
- logistic.py -> `LogisticRegression`
- knn.py -> `KNN`
- naive_bayes.py -> `GaussianNaiveBayes`
- decision_tree.py -> `DecisionTreeClassifier`
- svm.py -> `SupportVectorMachine`
- pipeline helpers

Each utility should follow a minimal, well-documented API.
"""

from .logistic import LogisticRegression
from .knn import KNearestNeighbors
from .naive_bayes import GaussianNaiveBayes
from .decision_tree import DecisionTreeClassifier
from .svm import SupportVectorMachine

__all__ = ["LogisticRegression", "DecisionTreeClassifier", "KNearestNeighbors", "GaussianNaiveBayes", "SupportVectorMachine"]
