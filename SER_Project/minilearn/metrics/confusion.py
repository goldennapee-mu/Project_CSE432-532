"""
Confusion matrix utilities module.

Implements confusion matrix computation for evaluating classification performance.
"""

import numpy as np
from typing import Optional


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                     labels: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute confusion matrix.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth labels.
    y_pred : array-like, shape (n_samples,)
        Predicted labels.
    labels : array-like, optional
        Unique class labels. If None, inferred from y_true and y_pred.
        
    Returns
    -------
    cm : ndarray, shape (n_classes, n_classes)
        Confusion matrix where cm[i, j] is the count of observations
        known to be group i but predicted to be group j.
        
    Notes
    -----
    - Rows represent true labels, columns represent predicted labels.
    - Diagonal elements are correct predictions.
    - Off-diagonal elements are misclassifications.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    # Create mapping from label to index
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1
    
    return cm

