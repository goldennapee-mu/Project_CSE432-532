"""
MiniLearn metrics package.

Implement metric utilities here:
- classification.py -> `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
- confusion.py -> `confusion_matrix` and helpers for visualization
- clustering.py -> `adjusted_rand_index`, `normalized_mutual_information`
- pipeline helpers

Each utility should follow a minimal, well-documented API.
"""

from .classification import accuracy_score, precision_score, recall_score, f1_score
from .confusion import confusion_matrix
from .clustering import adjusted_rand_index, normalized_mutual_information

__all__ = [
    'accuracy_score', 'precision_score', 'recall_score', 'f1_score',
    'confusion_matrix',
    'adjusted_rand_index', 'normalized_mutual_information',
]
