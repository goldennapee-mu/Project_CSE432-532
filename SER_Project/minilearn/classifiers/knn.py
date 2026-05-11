"""
k-nearest neighbors classifier module.

This module implements a k-nearest neighbors classifier following scikit-learn conventions.
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Optional

class KNearestNeighbors:
    """k-Nearest Neighbors classifier using majority vote.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : {'uniform', 'distance'}, default='uniform'
        If 'uniform', all neighbors are weighted equally.
        If 'distance', neighbors are weighted by inverse distance.
    metric : {'euclidean'}, default='euclidean'
        Distance metric to use.
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform',
                 metric: str = 'euclidean') -> None:
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if weights not in ('uniform', 'distance'):
            raise ValueError("weights must be 'uniform' or 'distance'")
        if metric != 'euclidean':
            raise ValueError("Only 'euclidean' metric is supported")
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self._x_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
    
    def fit(self, X, y):
        """Store training data (KNN is lazy learner).
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features.
        y : array-like, shape (n_samples,)
            Training labels.
        """
        self._x_train = np.asarray(X, dtype=np.float64)
        self._y_train = np.asarray(y)
        return self
    
    def _euclidean_distance(self, x1, x2):
        """Compute Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _find_neighbors(self, x):
        """Find k nearest neighbors and their distances.
        
        Returns
        -------
        neighbors : array, shape (n_neighbors,)
            Labels of k nearest neighbors.
        distances : array, shape (n_neighbors,)
            Distances to k nearest neighbors.
        """
        if self._x_train is None or self._y_train is None:
            raise ValueError("Model must be fitted first")
        
        distances = np.array(
            [self._euclidean_distance(x, x_train) for x_train in self._x_train]
        )
        k_indices = np.argsort(distances)[: self.n_neighbors]
        return self._y_train[k_indices], distances[k_indices]
    
    def _predict_single(self, x):
        """Predict class for a single sample."""
        neighbors, distances = self._find_neighbors(x)
        
        if self.weights == 'uniform':
            most_common = Counter(neighbors).most_common(1)
            return most_common[0][0]
        else:  # weights == 'distance'
            # Avoid division by zero for neighbors at distance 0
            weights = np.where(distances == 0, 1.0, 1.0 / distances)
            weighted_votes = {}
            for neighbor, weight in zip(neighbors, weights):
                key = neighbor
                weighted_votes[key] = weighted_votes.get(key, 0) + float(weight)
            return max(weighted_votes, key=lambda k: weighted_votes[k])
    
    def predict(self, X):
        """Predict class labels.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Query features.
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted labels.
        """
        if self._x_train is None:
            raise ValueError("Model must be fitted before calling predict")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        return np.array([self._predict_single(x) for x in X])
    
    def score(self, X, y):
        """Compute accuracy score.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test features.
        y : array-like, shape (n_samples,)
            True labels.
            
        Returns
        -------
        accuracy : float
            Fraction of correct predictions.
        """
        y_pred = self.predict(X)
        y = np.asarray(y)
        return np.mean(y_pred == y)
