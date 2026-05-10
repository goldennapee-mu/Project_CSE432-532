"""
Gaussian Naive Bayes classifier module.

This module implements Gaussian Naive Bayes following scikit-learn conventions.
"""

import numpy as np

class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier.
    
    Assumes each feature is normally distributed per class.
    Uses Bayes theorem to compute posterior probabilities.
    """
    
    def __init__(self, var_smoothing: float = 1e-9) -> None:
        """Initialize Gaussian Naive Bayes.
        
        Parameters
        ----------
        var_smoothing : float, default=1e-9
            Small value to stabilize variance estimates.
        """
        self.var_smoothing = var_smoothing
        self._class_priors: dict = {}
        self._class_means: dict = {}
        self._class_vars: dict = {}
        self._classes = None
    
    def fit(self, X, y):
        """Fit Gaussian Naive Bayes by computing class statistics.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features.
        y : array-like, shape (n_samples,)
            Training labels.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        self._classes = np.unique(y)
        n_samples = X.shape[0]
        
        for cls in self._classes:
            x_cls = X[y == cls]
            self._class_priors[cls] = x_cls.shape[0] / n_samples
            self._class_means[cls] = np.mean(x_cls, axis=0)
            self._class_vars[cls] = np.var(x_cls, axis=0) + self.var_smoothing
        
        return self
    
    def _calculate_likelihood(self, x, cls):
        """Calculate likelihood P(x|class) using normal distribution."""
        mean = self._class_means[cls]
        var = self._class_vars[cls]
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def _calculate_posterior(self, x):
        """Calculate posterior probabilities for each class."""
        if self._classes is None:
            raise ValueError("Model must be fitted first")
        
        posteriors = {}
        for cls in self._classes:
            prior = self._class_priors[cls]
            likelihood = self._calculate_likelihood(x, cls)
            posteriors[cls] = prior * np.prod(likelihood)
        return posteriors
    
    def _get_max_posterior(self, posteriors):
        """Return class with maximum posterior probability."""
        return max(posteriors, key=lambda k: posteriors[k])
    
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
        for x in X:
            posteriors = self._calculate_posterior(x)
            pred = self._get_max_posterior(posteriors)
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
