"""
MiniLearn — A minimal scikit-learn-style machine learning library.

Built from scratch for educational purposes as part of the CSE 432/532
Speech Emotion Recognition project.

Usage:
    from minilearn.classifiers import LogisticRegression, KNN, GaussianNaiveBayes, DecisionTreeClassifier
    from minilearn.preprocessing import StandardScaler, train_test_split
    from minilearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
"""

# Export subpackages for easier imports. Implementations live in the
# `classifiers`, `preprocessing`, `metrics`, and `model_selection` modules.
from . import classifiers, preprocessing, metrics, model_selection

__version__ = "0.1.0"

__all__ = ["classifiers", "preprocessing", "metrics", "model_selection", "__version__"]
