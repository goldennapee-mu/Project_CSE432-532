"""
Decision tree classifier module.

This implementation uses a small CART-style tree with Gini impurity and binary
threshold splits so it can handle the continuous SER feature table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class _TreeNode:
	"""Single node in the decision tree."""

	feature_index: int | None = None
	threshold: float | None = None
	left: "_TreeNode | None" = None
	right: "_TreeNode | None" = None
	value: object | None = None

class DecisionTreeClassifier:
	"""A small decision tree classifier built from scratch.

	Parameters
	----------
	max_depth:
		Maximum tree depth. ``None`` means no explicit depth limit.
	min_samples_split:
		Minimum number of samples required to split an internal node.
	min_samples_leaf:
		Minimum number of samples allowed in a leaf node.
	min_impurity_decrease:
		Minimum impurity improvement required to accept a split.
	random_state:
		Optional seed used for deterministic tie-breaking.
	"""

	def __init__(
		self,
		max_depth: int | None = None,
		min_samples_split: int = 2,
		min_samples_leaf: int = 1,
		min_impurity_decrease: float = 0.0,
		random_state: int | None = None,
	) -> None:
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.min_impurity_decrease = min_impurity_decrease
		self.random_state = random_state

	def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
		"""Fit the tree on training data."""
		X = np.asarray(X, dtype=float)
		y = np.asarray(y)

		if X.ndim != 2:
			raise ValueError("X must be a 2D array")
		if y.ndim != 1:
			raise ValueError("y must be a 1D array")
		if X.shape[0] != y.shape[0]:
			raise ValueError("X and y must contain the same number of rows")

		self.classes_ = np.unique(y)
		self.n_classes_ = self.classes_.shape[0]
		self.n_features_in_ = X.shape[1]
		self._class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
		self._rng = np.random.default_rng(self.random_state)
		self._y_indices = np.array([self._class_to_index[label] for label in y], dtype=int)
		self.root_ = self._build_tree(X, self._y_indices, depth=0)
		return self

	def _gini_from_counts(self, class_counts: np.ndarray) -> float:
		total = class_counts.sum()
		if total == 0:
			return 0.0
		probabilities = class_counts / total
		return float(1.0 - np.sum(probabilities * probabilities))

	def _majority_class(self, y_indices: np.ndarray) -> object:
		counts = np.bincount(y_indices, minlength=self.n_classes_)
		winners = np.flatnonzero(counts == counts.max())
		winner = int(self._rng.choice(winners)) if winners.size > 1 else int(winners[0])
		return self.classes_[winner]

	def _evaluate_feature_split(
		self,
		feature_values: np.ndarray,
		y_indices: np.ndarray,
		parent_gini: float,
	) -> tuple[float, float | None, np.ndarray | None]:
		"""Evaluate one feature and return its best impurity improvement."""
		n_samples = y_indices.shape[0]
		order = np.argsort(feature_values, kind="mergesort")
		sorted_x = feature_values[order]
		sorted_y = y_indices[order]

		if np.all(sorted_x == sorted_x[0]):
			return 0.0, None, None

		left_counts = np.zeros(self.n_classes_, dtype=int)
		right_counts = np.bincount(sorted_y, minlength=self.n_classes_)

		best_impurity_decrease = 0.0
		best_threshold: float | None = None
		best_mask: np.ndarray | None = None

		for split_index in range(1, n_samples):
			class_index = sorted_y[split_index - 1]
			left_counts[class_index] += 1
			right_counts[class_index] -= 1

			if sorted_x[split_index] == sorted_x[split_index - 1]:
				continue

			left_size = split_index
			right_size = n_samples - split_index
			if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
				continue

			threshold = (sorted_x[split_index] + sorted_x[split_index - 1]) / 2.0
			left_gini = self._gini_from_counts(left_counts)
			right_gini = self._gini_from_counts(right_counts)
			weighted_gini = (left_size / n_samples) * left_gini + (right_size / n_samples) * right_gini
			impurity_decrease = parent_gini - weighted_gini

			if impurity_decrease > best_impurity_decrease:
				best_impurity_decrease = impurity_decrease
				best_threshold = float(threshold)
				best_mask = feature_values <= threshold

		return best_impurity_decrease, best_threshold, best_mask

	def _best_split(self, X: np.ndarray, y_indices: np.ndarray) -> tuple[int | None, float | None, np.ndarray | None]:
		_, n_features = X.shape
		parent_counts = np.bincount(y_indices, minlength=self.n_classes_)
		parent_gini = self._gini_from_counts(parent_counts)

		best_feature: int | None = None
		best_threshold: float | None = None
		best_mask: np.ndarray | None = None
		best_impurity_decrease = 0.0

		for feature_index in range(n_features):
			feature_values = X[:, feature_index]
			feature_impurity_decrease, feature_threshold, feature_mask = self._evaluate_feature_split(
				feature_values,
				y_indices,
				parent_gini,
			)

			if feature_impurity_decrease > best_impurity_decrease:
				best_impurity_decrease = feature_impurity_decrease
				best_feature = feature_index
				best_threshold = feature_threshold
				best_mask = feature_mask

		if best_impurity_decrease < self.min_impurity_decrease:
			return None, None, None

		return best_feature, best_threshold, best_mask

	def _build_tree(self, X: np.ndarray, y_indices: np.ndarray, depth: int) -> _TreeNode:
		node = _TreeNode(value=self._majority_class(y_indices))

		if y_indices.size < self.min_samples_split:
			return node
		if np.unique(y_indices).size == 1:
			return node
		if self.max_depth is not None and depth >= self.max_depth:
			return node

		feature_index, threshold, mask = self._best_split(X, y_indices)
		if feature_index is None or threshold is None or mask is None:
			return node

		node.feature_index = feature_index
		node.threshold = threshold
		node.left = self._build_tree(X[mask], y_indices[mask], depth + 1)
		node.right = self._build_tree(X[~mask], y_indices[~mask], depth + 1)
		return node

	def _predict_one(self, row: np.ndarray) -> object:
		node = self.root_
		while node.feature_index is not None and node.threshold is not None:
			if row[node.feature_index] <= node.threshold:
				node = node.left
			else:
				node = node.right
			if node is None:
				break
		return node.value if node is not None else self.classes_[0]

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""Predict class labels for rows in X."""
		if not hasattr(self, "root_"):
			raise ValueError("This DecisionTreeClassifier instance is not fitted yet")

		X = np.asarray(X, dtype=float)
		if X.ndim != 2:
			raise ValueError("X must be a 2D array")
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but this model was fitted with {self.n_features_in_}"
			)

		return np.array([self._predict_one(row) for row in X], dtype=object)

	def score(self, X: np.ndarray, y: np.ndarray) -> float:
		"""Return mean classification accuracy."""
		y_true = np.asarray(y)
		y_pred = self.predict(X)
		return float(np.mean(y_pred == y_true))
