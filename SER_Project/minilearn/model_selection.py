"""
K-Fold Cross-Validation utilities module.

Implements stratified k-fold cross-validation for model evaluation and hyperparameter tuning.
"""

from __future__ import annotations

import numpy as np
from typing import Iterator, Optional, Tuple

class StratifiedKFold:
    """Stratified K-Fold cross-validator.
    
    Splits dataset into k folds while preserving class distribution in each fold.
    Useful for imbalanced datasets to ensure consistent class representation.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: Optional[int] = None) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test sets.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        y : array-like, shape (n_samples,)
            Target labels.
        groups : array-like, optional
            Group labels for the samples. Not used, present for API compatibility.
            
        Yields
        ------
        train_idx : array, shape (n_train,)
            Training set indices.
        test_idx : array, shape (n_test,)
            Test set indices.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        
        # Get unique classes and their indices
        classes, class_counts = np.unique(y, return_counts=True)
        
        # Initialize random number generator
        rng = np.random.default_rng(self.random_state)
        
        # Create fold assignments
        fold_assignments = np.zeros(n_samples, dtype=int)
        
        for cls, count in zip(classes, class_counts):
            cls_indices = np.nonzero(y == cls)[0]
            
            if self.shuffle:
                rng.shuffle(cls_indices)
            
            # Distribute class samples across folds
            fold_size = count / self.n_splits
            for i, idx in enumerate(cls_indices):
                fold_assignments[idx] = int(i / fold_size)
        
        # Generate train/test splits
        for fold in range(self.n_splits):
            test_idx = np.nonzero(fold_assignments == fold)[0]
            train_idx = np.nonzero(fold_assignments != fold)[0]
            
            yield train_idx, test_idx
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return number of splits (API compatible with scikit-learn)."""
        return self.n_splits

def cross_validate(model, X: np.ndarray, y: np.ndarray, cv=None,
                   scoring=None) -> dict:
    """Perform cross-validation on model.
    
    Parameters
    ----------
    model : estimator
        Estimator with fit() and score() methods.
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target labels.
    cv : int or KFold-like, default=None
        Number of folds (default=5) or cross-validator object.
    scoring : callable, optional
        Scoring function. If None, uses model.score().
        
    Returns
    -------
    scores : dict
        Dictionary with 'test_scores' (array of test fold scores) and
        'train_scores' (array of training fold scores).
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5)
    elif isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv)
    
    test_scores = []
    train_scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Score
        if scoring is not None:
            test_score = scoring(y_test, model.predict(X_test))
            train_score = scoring(y_train, model.predict(X_train))
        else:
            test_score = model.score(X_test, y_test)
            train_score = model.score(X_train, y_train)
        
        test_scores.append(test_score)
        train_scores.append(train_score)
    
    return {
        'test_scores': np.array(test_scores),
        'train_scores': np.array(train_scores)
    }
