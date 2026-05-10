"""
Classification metrics module.

Implements evaluation metrics for classification tasks:
- accuracy_score: Proportion of correct predictions
- precision_score: True positives / (true positives + false positives)
- recall_score: True positives / (true positives + false negatives)
- f1_score: Harmonic mean of precision and recall
"""

import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth labels.
    y_pred : array-like, shape (n_samples,)
        Predicted labels.
        
    Returns
    -------
    accuracy : float
        Fraction of correct predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, 
                   average: str = 'macro', 
                   zero_division: int = 0) -> float:
    """Compute precision score.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth labels.
    y_pred : array-like, shape (n_samples,)
        Predicted labels.
    average : {'macro', 'micro', 'weighted'}, default='macro'
        Averaging method for multiclass:
        - 'macro': unweighted mean of precision for each class
        - 'micro': global precision (same as accuracy for multiclass)
        - 'weighted': weighted by class frequency
    zero_division : int, default=0
        Value to return when no predictions for a class.
        
    Returns
    -------
    precision : float
        Precision score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(y_true)
    precisions = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        
        if tp + fp == 0:
            precisions.append(zero_division)
        else:
            precisions.append(tp / (tp + fp))
    
    if average == 'macro':
        return float(np.mean(precisions))
    elif average == 'micro':
        return accuracy_score(y_true, y_pred)
    elif average == 'weighted':
        class_counts = np.bincount(y_true, minlength=len(classes))
        weights = class_counts / len(y_true)
        return float(np.sum(np.array(precisions) * weights))
    else:
        raise ValueError(f"average must be 'macro', 'micro', or 'weighted', got {average}")


def recall_score(y_true: np.ndarray, y_pred: np.ndarray,
                average: str = 'macro',
                zero_division: int = 0) -> float:
    """Compute recall score.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth labels.
    y_pred : array-like, shape (n_samples,)
        Predicted labels.
    average : {'macro', 'micro', 'weighted'}, default='macro'
        Averaging method for multiclass:
        - 'macro': unweighted mean of recall for each class
        - 'micro': global recall (same as accuracy for multiclass)
        - 'weighted': weighted by class frequency
    zero_division : int, default=0
        Value to return when no instances of a class.
        
    Returns
    -------
    recall : float
        Recall score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(y_true)
    recalls = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        
        if tp + fn == 0:
            recalls.append(zero_division)
        else:
            recalls.append(tp / (tp + fn))
    
    if average == 'macro':
        return float(np.mean(recalls))
    elif average == 'micro':
        return accuracy_score(y_true, y_pred)
    elif average == 'weighted':
        class_counts = np.bincount(y_true, minlength=len(classes))
        weights = class_counts / len(y_true)
        return float(np.sum(np.array(recalls) * weights))
    else:
        raise ValueError(f"average must be 'macro', 'micro', or 'weighted', got {average}")


def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
            average: str = 'macro',
            zero_division: int = 0) -> float:
    """Compute F1 score.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth labels.
    y_pred : array-like, shape (n_samples,)
        Predicted labels.
    average : {'macro', 'micro', 'weighted'}, default='macro'
        Averaging method for multiclass:
        - 'macro': unweighted mean of F1 for each class
        - 'micro': global F1 (same as accuracy for multiclass)
        - 'weighted': weighted by class frequency
    zero_division : int, default=0
        Value to return when no predictions for a class.
        
    Returns
    -------
    f1 : float
        F1 score (harmonic mean of precision and recall).
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, average=average, zero_division=zero_division)
    
    if precision + recall == 0:
        return zero_division
    
    return float(2 * (precision * recall) / (precision + recall))
