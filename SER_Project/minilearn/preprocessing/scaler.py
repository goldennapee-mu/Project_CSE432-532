"""
Feature scaling utilities module.

Implements StandardScaler for feature normalization.
"""

from __future__ import annotations

import numpy as np

class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance.
    
    Computes mean and standard deviation from training data, then applies
    the same transformation to any data (training or test).
    
    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.
        
    Attributes
    ----------
    mean_ : ndarray, shape (n_features,)
        Per feature mean computed during fit.
    scale_ : ndarray, shape (n_features,)
        Per feature standard deviation computed during fit.
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray, y = None) -> 'StandardScaler':
        """Compute mean and standard deviation for later scaling.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ignored
            Ignored. Present for scikit-learn API compatibility.
            
        Returns
        -------
        self : StandardScaler
            Fitted scaler.
        """
        X = np.asarray(X)
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
        
        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            # Avoid division by zero
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.scale_ = np.ones(X.shape[1])
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted mean and standard deviation.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_features)
            Standardized data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        X = np.asarray(X)
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X: np.ndarray, y = None) -> np.ndarray:
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ignored
            Ignored. Present for scikit-learn API compatibility.
            
        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_features)
            Standardized data.
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Scale back the data to the original representation.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Standardized data.
            
        Returns
        -------
        X_original : ndarray, shape (n_samples, n_features)
            Original-scale data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        X = np.asarray(X)
        return X * self.scale_ + self.mean_
