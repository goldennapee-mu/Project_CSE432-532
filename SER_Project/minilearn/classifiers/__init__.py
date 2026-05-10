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

from . import knn, naive_bayes
from .decision_tree import DecisionTreeClassifier
from .logistic import LogisticRegression

__all__ = ["LogisticRegression", "DecisionTreeClassifier", "knn", "naive_bayes"]
