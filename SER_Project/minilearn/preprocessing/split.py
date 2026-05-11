"""
Data splitting utilities module.

Implements train_test_split for partitioning datasets.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

def train_test_split(X: np.ndarray, y: Optional[np.ndarray] = None, 
                    test_size: float = 0.2, random_state: Optional[int] = None,
                    stratify: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Split arrays into random train and test subsets.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features) or (n_samples,)
        Data to split.
    y : ndarray, shape (n_samples,), optional
        Labels corresponding to X. If provided, will be split as well.
    test_size : float, default=0.2
        Proportion of dataset to include in test split (between 0 and 1).
    random_state : int or None, default=None
        Random seed for reproducibility.
    stratify : ndarray, shape (n_samples,), optional
        If provided, perform stratified split to preserve class distribution.
        Typically the same as y.
        
    Returns
    -------
    X_train : ndarray
        Training data.
    X_test : ndarray
        Test data.
    y_train : ndarray or None
        Training labels (None if y was not provided).
    y_test : ndarray or None
        Test labels (None if y was not provided).
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    n_test = int(np.ceil(n_samples * test_size))
    n_train = n_samples - n_test
    
    rng = np.random.default_rng(random_state)
    
    if stratify is not None:
        # Stratified split: preserve class distribution
        stratify = np.asarray(stratify)
        classes = np.unique(stratify)
        
        train_indices = []
        test_indices = []
        
        for cls in classes:
            cls_indices = np.nonzero(stratify == cls)[0]
            cls_n_test = max(1, int(np.ceil(len(cls_indices) * test_size)))
            
            # Shuffle indices within class
            rng.shuffle(cls_indices)
            
            test_indices.extend(cls_indices[:cls_n_test])
            train_indices.extend(cls_indices[cls_n_test:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
    else:
        # Random split
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    y_train = y[train_indices] if y is not None else None
    y_test = y[test_indices] if y is not None else None
    
    return X_train, X_test, y_train, y_test
