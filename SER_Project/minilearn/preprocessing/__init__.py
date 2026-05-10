"""
MiniLearn preprocessing package.

Implement preprocessing utilities here:
- scaler.py -> `StandardScaler`
- split.py -> `train_test_split`
- pca.py -> `PCA`
- pipeline helpers

Each utility should follow a minimal, well-documented API.
"""

from .scaler import StandardScaler
from .split import train_test_split
from .pca import PCA

__all__ = ["StandardScaler", "train_test_split", "PCA"]
