"""
Support Vector Machine (SVM) classifier module.

This module implements a linear SVM using hinge loss and gradient descent,
supporting multi-class classification via One-vs-Rest strategy.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

class SupportVectorMachine:
    """Linear SVM classifier using hinge loss and gradient descent.
    
    Uses One-vs-Rest strategy for multi-class classification.
    
    Parameters
    ----------
    c_param : float, default=1.0
        Regularization parameter (inverse of penalty).
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    max_iter : int, default=500
        Maximum number of iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(self, c_param: float = 1.0, learning_rate: float = 0.01,
                 max_iter: int = 500, tol: float = 1e-4,
                 random_state: Optional[int] = None) -> None:
        if c_param <= 0:
            raise ValueError("c_param must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        
        self.c_param = c_param
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self._classifiers: dict = {}
        self._classes = None
        self._scaler_mean = None
        self._scaler_std = None
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean and unit variance."""
        if self._scaler_mean is None or self._scaler_std is None:
            raise ValueError("Model must be fitted before normalizing features")
        return (X - self._scaler_mean) / (self._scaler_std + 1e-8)
    
    def _fit_binary_svm(self, X: np.ndarray, y: np.ndarray,
                        pos_class) -> tuple:
        """Fit binary SVM for one class vs rest."""
        # Convert to binary labels
        y_binary = np.where(y == pos_class, 1, -1)
        
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        for _ in range(self.max_iter):
            # Compute margins
            margins = y_binary * (X @ w + b)
            
            # Hinge loss: max(0, 1 - margin)
            losses = np.maximum(0, 1 - margins)
            
            # Check convergence
            if np.mean(losses) < self.tol:
                break
            
            # Gradient computation
            mask = losses > 0
            dw = np.zeros(n_features)
            db = 0.0
            
            for i in range(n_samples):
                if mask[i]:
                    dw -= y_binary[i] * X[i]
                    db -= y_binary[i]
            
            # Add regularization term: (1/2) * ||w||^2
            dw += (1 / self.c_param) * w
            
            # Gradient descent update
            w -= self.learning_rate * dw / n_samples
            b -= self.learning_rate * db / n_samples
        
        return w, b
    
    def fit(self, X, y):
        """Fit SVM using One-vs-Rest strategy.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features.
        y : array-like, shape (n_samples,)
            Training labels.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # Normalize features
        self._scaler_mean = np.mean(X, axis=0)
        self._scaler_std = np.std(X, axis=0)
        x_norm = self._normalize_features(X)
        
        self._classes = np.unique(y)
        
        # Fit binary SVM for each class
        for cls in self._classes:
            w, b = self._fit_binary_svm(x_norm, y, cls)
            self._classifiers[cls] = (w, b)
        
        return self
    
    def _predict_scores(self, X: np.ndarray) -> dict:
        """Compute decision scores for each class."""
        if self._classes is None:
            raise ValueError("Model must be fitted before calling predict")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        x_norm = self._normalize_features(X)
        scores = {}
        
        for cls in self._classes:
            w, b = self._classifiers[cls]
            scores[cls] = x_norm @ w + b
        
        return scores
    
    def _get_max_class(self, scores: dict):
        """Return class with maximum score."""
        return max(scores, key=lambda k: scores[k])
    
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
        if self._classes is None:
            raise ValueError("Model must be fitted before calling predict")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for i in range(X.shape[0]):
            x_sample = X[i:i+1]
            scores = self._predict_scores(x_sample)
            pred = self._get_max_class(scores)
            predictions.append(pred)
        
        return np.array(predictions)
    
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
