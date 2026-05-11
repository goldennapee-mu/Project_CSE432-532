"""
Clustering evaluation metrics.

Implements metrics for evaluating clustering quality:
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
"""

from __future__ import annotations

import numpy as np

def adjusted_rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Adjusted Rand Index (ARI).
    
    ARI measures the similarity between true labels and predicted clusters,
    adjusted for chance. Range: [-1, 1] where 1 is perfect agreement,
    0 is random clustering, negative values indicate worse than random.
    
    Parameters
    ----------
    y_true : ndarray, shape (n_samples,)
        Ground truth labels.
    y_pred : ndarray, shape (n_samples,)
        Predicted cluster labels.
        
    Returns
    -------
    ari : float
        Adjusted Rand Index between -1 and 1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n_samples = len(y_true)
    
    # Get unique labels
    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)
    
    n_true_classes = len(true_classes)
    n_pred_classes = len(pred_classes)
    
    # Create contingency matrix (confusion matrix)
    contingency = np.zeros((n_true_classes, n_pred_classes), dtype=np.int64)
    
    true_class_to_idx = {cls: idx for idx, cls in enumerate(true_classes)}
    pred_class_to_idx = {cls: idx for idx, cls in enumerate(pred_classes)}
    
    for t, p in zip(y_true, y_pred):
        t_idx = true_class_to_idx[t]
        p_idx = pred_class_to_idx[p]
        contingency[t_idx, p_idx] += 1
    
    # Compute Rand Index
    # a = pairs that are both same in true and pred
    # b = pairs that are both different in true and pred
    
    # Sum of squares of contingency matrix elements
    sum_cij2 = np.sum(contingency ** 2)
    
    # Sum of squares of row sums
    row_sums = np.sum(contingency, axis=1)
    sum_ri2 = np.sum(row_sums ** 2)
    
    # Sum of squares of column sums
    col_sums = np.sum(contingency, axis=0)
    sum_cj2 = np.sum(col_sums ** 2)
    
    # Rand Index components
    n_choose_2 = n_samples * (n_samples - 1) / 2
    
    # Number of pairs in same class and same cluster
    a = (sum_cij2 - n_samples) / 2
    
    # Expected number of pairs in same class and same cluster (under random clustering)
    expected = ((sum_ri2 - n_samples) * (sum_cj2 - n_samples)) / (2 * n_choose_2)
    
    # Maximum possible value
    max_val = ((sum_ri2 - n_samples) + (sum_cj2 - n_samples)) / 2
    
    # Adjusted Rand Index
    if max_val == expected:
        return 1.0 if a == expected else 0.0
    
    ari = (a - expected) / (max_val - expected)
    return float(ari)


def normalized_mutual_information(y_true: np.ndarray, y_pred: np.ndarray, average_method: str = 'arithmetic') -> float:
    """Compute Normalized Mutual Information (NMI).
    
    NMI measures the mutual dependence between true labels and predicted
    clusters, normalized by entropy. Range: [0, 1] where 1 is perfect agreement,
    0 is independence.
    
    Parameters
    ----------
    y_true : ndarray, shape (n_samples,)
        Ground truth labels.
    y_pred : ndarray, shape (n_samples,)
        Predicted cluster labels.
    average_method : {'arithmetic', 'geometric', 'min', 'max'}, default='arithmetic'
        Method to normalize mutual information.
        
    Returns
    -------
    nmi : float
        Normalized Mutual Information between 0 and 1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n_samples = len(y_true)
    
    # Get unique labels
    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)
    
    n_true_classes = len(true_classes)
    n_pred_classes = len(pred_classes)
    
    # Create contingency matrix
    contingency = np.zeros((n_true_classes, n_pred_classes), dtype=np.int64)
    
    true_class_to_idx = {cls: idx for idx, cls in enumerate(true_classes)}
    pred_class_to_idx = {cls: idx for idx, cls in enumerate(pred_classes)}
    
    for t, p in zip(y_true, y_pred):
        t_idx = true_class_to_idx[t]
        p_idx = pred_class_to_idx[p]
        contingency[t_idx, p_idx] += 1
    
    # Compute marginal probabilities
    pi = np.sum(contingency, axis=1) / n_samples  # p(true class)
    pj = np.sum(contingency, axis=0) / n_samples  # p(pred class)
    pij = contingency / n_samples                  # p(true class, pred class)
    
    # Compute entropies
    # H(true) = -sum(pi * log(pi))
    h_true = -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
    
    # H(pred) = -sum(pj * log(pj))
    h_pred = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))
    
    # Mutual Information I(true; pred) = sum(pij * log(pij / (pi * pj)))
    mi = 0.0
    for i in range(n_true_classes):
        for j in range(n_pred_classes):
            if pij[i, j] > 0:
                mi += pij[i, j] * np.log(pij[i, j] / (pi[i] * pj[j]))
    
    # Normalize by average of entropies
    if average_method == 'arithmetic':
        norm = (h_true + h_pred) / 2
    elif average_method == 'geometric':
        norm = np.sqrt(h_true * h_pred)
    elif average_method == 'min':
        norm = min(h_true, h_pred)
    elif average_method == 'max':
        norm = max(h_true, h_pred)
    else:
        raise ValueError(f"average_method must be 'arithmetic', 'geometric', 'min', or 'max', got {average_method}")
    
    if norm == 0:
        return 0.0
    
    nmi = mi / norm
    return float(nmi)
