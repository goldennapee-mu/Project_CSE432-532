"""
Principal Component Analysis (PCA) for dimensionality reduction.

Implements PCA via eigendecomposition of the covariance matrix.
"""

import numpy as np
from typing import Optional


class PCA:
    """Principal Component Analysis (PCA).
    
    Performs linear dimensionality reduction using Singular Value Decomposition
    of the data to project it to a lower dimensional space.
    
    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to keep. If None, keep all.
    whiten : bool, default=False
        If True, scale components by singular values (for preprocessing).
        
    Attributes
    ----------
    components_ : ndarray, shape (n_components, n_features)
        Principal axes in feature space.
    explained_variance_ : ndarray, shape (n_components,)
        Variance explained by each component.
    explained_variance_ratio_ : ndarray, shape (n_components,)
        Proportion of variance explained by each component.
    mean_ : ndarray, shape (n_features,)
        Per feature mean computed during fit.
    n_features_in_ : int
        Number of features seen during fit.
    """
    
    def __init__(self, n_components: Optional[int] = None, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten
        
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_features_in_ = None
        self._singular_values = None
    
    def fit(self, X: np.ndarray, y = None) -> 'PCA':
        """Fit PCA model to data.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ignored
            Ignored. Present for scikit-learn API compatibility.
            
        Returns
        -------
        self : PCA
            Fitted PCA model.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute SVD
        # U: (n_samples, min(n_samples, n_features))
        # S: (min(n_samples, n_features),)
        # Vt: (min(n_samples, n_features), n_features)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Components are rows of Vt (transposed to get columns as components)
        # Total possible components
        n_components_all = len(S)
        
        if self.n_components is None:
            n_components = n_components_all
        else:
            n_components = min(self.n_components, n_components_all)
        
        # Store components
        self.components_ = Vt[:n_components]
        
        # Compute explained variance
        explained_variance = (S ** 2) / (n_samples - 1)
        self.explained_variance_ = explained_variance[:n_components]
        
        # Compute explained variance ratio
        total_variance = np.sum(explained_variance)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        self._singular_values = S[:n_components]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto principal components.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_components)
            Transformed data in lower dimensional space.
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        X = np.asarray(X)
        
        # Center data using training mean
        X_centered = X - self.mean_
        
        # Project onto components
        X_transformed = X_centered @ self.components_.T
        
        if self.whiten and self.explained_variance_ is not None:
            X_transformed = X_transformed / np.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray, y = None) -> np.ndarray:
        """Fit PCA and transform data in one step.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ignored
            Ignored. Present for scikit-learn API compatibility.
            
        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Transform data back to original feature space.
        
        Parameters
        ----------
        X_transformed : ndarray, shape (n_samples, n_components)
            Transformed data in reduced space.
            
        Returns
        -------
        X_original : ndarray, shape (n_samples, n_features)
            Data in original feature space.
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        X_transformed = np.asarray(X_transformed)
        
        if self.whiten and self.explained_variance_ is not None:
            X_transformed = X_transformed * np.sqrt(self.explained_variance_)
        
        # Project back to original space
        X_original = X_transformed @ self.components_ + self.mean_
        
        return X_original
