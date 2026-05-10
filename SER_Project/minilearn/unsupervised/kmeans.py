"""
K-Means clustering implementation.

Implements unsupervised clustering using the K-Means algorithm.
"""

import numpy as np
from typing import Optional, Tuple, cast

class KMeans:
    """K-Means clustering algorithm.
    
    Clusters data into k groups using iterative centroid updates and
    Euclidean distance.
    
    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    max_iter : int, default=300
        Maximum number of iterations.
    random_state : int or None, default=None
        Random seed for reproducible centroid initialization.
    tol : float, default=1e-4
        Convergence tolerance (change in centroids).
    """
    
    def __init__(self, n_clusters: int = 8, max_iter: int = 300, 
                 random_state: Optional[int] = None, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.inertia_history_ = []
    
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids randomly from data points.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        centroids : ndarray, shape (n_clusters, n_features)
            Initial centroids.
        """
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _euclidean_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between samples and centroids.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data points.
        centroids : ndarray, shape (n_clusters, n_features)
            Cluster centroids.
            
        Returns
        -------
        distances : ndarray, shape (n_samples, n_clusters)
            Distance from each sample to each centroid.
        """
        # Shape: (n_samples, 1, n_features) - (1, n_clusters, n_features)
        # Result: (n_samples, n_clusters)
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each sample to nearest centroid.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data points.
        centroids : ndarray, shape (n_clusters, n_features)
            Cluster centroids.
            
        Returns
        -------
        labels : ndarray, shape (n_samples,)
            Cluster assignment for each sample.
        """
        distances = self._euclidean_distance(X, centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """Update centroids as mean of assigned points.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data points.
        labels : ndarray, shape (n_samples,)
            Cluster assignments.
            
        Returns
        -------
        new_centroids : ndarray, shape (n_clusters, n_features)
            Updated centroids.
        centroid_shift : float
            Maximum L2 distance any centroid moved.
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids[k] = np.mean(X[mask], axis=0)
            else:
                # If cluster is empty, reinitialize to random point
                rng = np.random.default_rng(self.random_state)
                idx = rng.choice(X.shape[0])
                new_centroids[k] = X[idx]
        
        # Compute maximum shift
        if self.centroids_ is not None:
            centroid_shift = float(np.max(np.sqrt(np.sum((new_centroids - self.centroids_) ** 2, axis=1))))
        else:
            # First iteration: large shift to continue
            centroid_shift = float('inf')
        
        return new_centroids, centroid_shift
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Compute within-cluster sum of squares (inertia).
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data points.
        labels : ndarray, shape (n_samples,)
            Cluster assignments.
        centroids : ndarray, shape (n_clusters, n_features)
            Cluster centroids.
            
        Returns
        -------
        inertia : float
            Within-cluster sum of squares.
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                points = X[mask]
                distances = np.sum((points - centroids[k]) ** 2, axis=1)
                inertia += np.sum(distances)
        return inertia
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'KMeans':
        """Fit K-Means to data.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ignored
            Ignored. Present for scikit-learn API compatibility.
            
        Returns
        -------
        self : KMeans
            Fitted estimator.
        """
        X = np.asarray(X)
        
        # Initialize centroids
        self.centroids_ = self._init_centroids(X)
        self.inertia_history_ = []
        inertia = 0.0  # Initialize before loop
        
        for iteration in range(self.max_iter):
            # Assign clusters
            self.labels_ = self._assign_clusters(X, self.centroids_)
            
            # Compute inertia
            inertia = self._compute_inertia(X, self.labels_, self.centroids_)
            self.inertia_history_.append(inertia)
            
            # Update centroids
            new_centroids, centroid_shift = self._update_centroids(X, self.labels_)
            
            # Check convergence
            if centroid_shift < self.tol:
                self.n_iter_ = iteration + 1
                self.centroids_ = new_centroids
                self.inertia_ = inertia
                break
            
            self.centroids_ = new_centroids
            self.n_iter_ = iteration + 1
        
        self.inertia_ = inertia
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for samples.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        labels : ndarray, shape (n_samples,)
            Cluster label for each sample.
        """
        if self.centroids_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X)
        return self._assign_clusters(X, self.centroids_)
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit K-Means and return cluster labels.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ignored
            Ignored. Present for scikit-learn API compatibility.
            
        Returns
        -------
        labels : ndarray, shape (n_samples,)
            Cluster label for each sample.
        """
        self.fit(X, y)
        return cast(np.ndarray, self.labels_)
