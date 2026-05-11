"""
Logistic regression classifier module.

This implementation uses batch gradient descent and a softmax objective so it
can handle the multi-class SER labels in this project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

def _softmax(logits: np.ndarray) -> np.ndarray:
	"""Compute row-wise softmax in a numerically stable way."""
	shifted = logits - np.max(logits, axis=1, keepdims=True)
	exp_scores = np.exp(shifted)
	return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


@dataclass
class LogisticRegression:
	"""Softmax logistic regression for classification.

	Parameters
	----------
	learning_rate:
		Gradient descent step size.
	max_iter:
		Maximum number of optimization steps.
	tol:
		Early stopping threshold on the absolute change in loss.
	fit_intercept:
		Whether to add a bias term.
	l2_penalty:
		L2 regularization strength applied to weights, not the intercept.
	random_state:
		Optional seed for reproducible weight initialization.
	verbose:
		If true, prints the loss every 100 iterations.
	"""

	learning_rate: float = 0.1
	max_iter: int = 1000
	tol: float = 1e-6
	fit_intercept: bool = True
	l2_penalty: float = 0.0
	random_state: int | None = None
	verbose: bool = False

	def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Validate and normalize the inputs used during fitting."""
		X = np.asarray(X, dtype=float)
		y = np.asarray(y)

		if X.ndim != 2:
			raise ValueError("X must be a 2D array")
		if y.ndim != 1:
			raise ValueError("y must be a 1D array")
		if X.shape[0] != y.shape[0]:
			raise ValueError("X and y must contain the same number of rows")

		return X, y

	def _encode_labels(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Map labels to contiguous integer indices and one-hot targets."""
		self.classes_ = np.unique(y)
		self.n_classes_ = self.classes_.shape[0]
		class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
		y_indices = np.array([class_to_index[label] for label in y], dtype=int)
		y_one_hot = np.eye(self.n_classes_, dtype=float)[y_indices]
		return y_indices, y_one_hot

	def _augment_features(self, X: np.ndarray) -> np.ndarray:
		"""Add an intercept column when the model uses one."""
		if self.fit_intercept:
			return np.hstack([np.ones((X.shape[0], 1)), X])
		return X

	def _compute_loss(self, y_one_hot: np.ndarray, probabilities: np.ndarray) -> float:
		"""Compute cross-entropy loss with optional L2 regularization."""
		loss = -np.mean(np.sum(y_one_hot * np.log(probabilities + 1e-12), axis=1))
		if self.l2_penalty > 0:
			weights = self.coef_[1:, :] if self.fit_intercept else self.coef_
			loss += 0.5 * self.l2_penalty * np.sum(weights * weights)
		return loss

	def _compute_gradient(
		self,
		x_aug: np.ndarray,
		y_one_hot: np.ndarray,
		probabilities: np.ndarray,
	) -> np.ndarray:
		"""Compute the gradient of the softmax loss."""
		gradient = (x_aug.T @ (probabilities - y_one_hot)) / x_aug.shape[0]
		if self.l2_penalty > 0:
			if self.fit_intercept:
				gradient[1:, :] += self.l2_penalty * self.coef_[1:, :]
			else:
				gradient += self.l2_penalty * self.coef_
		return gradient

	def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
		"""Fit the model on training data."""
		X, y = self._validate_training_data(X, y)
		_, y_one_hot = self._encode_labels(y)
		self.n_features_in_ = X.shape[1]

		rng = np.random.default_rng(self.random_state)

		x_aug = self._augment_features(X)
		self.coef_ = rng.normal(scale=0.01, size=(x_aug.shape[1], self.n_classes_))

		previous_loss = np.inf

		for iteration in range(self.max_iter):
			logits = x_aug @ self.coef_
			probabilities = _softmax(logits)

			loss = self._compute_loss(y_one_hot, probabilities)
			gradient = self._compute_gradient(x_aug, y_one_hot, probabilities)

			self.coef_ -= self.learning_rate * gradient

			if self.verbose and iteration % 100 == 0:
				print(f"iteration={iteration} loss={loss:.6f}")

			if abs(previous_loss - loss) < self.tol:
				break
			previous_loss = loss

		return self

	def prepare_x(self, X: np.ndarray) -> np.ndarray:
		"""Validate and augment feature arrays."""
		if not hasattr(self, "coef_"):
			raise ValueError("This LogisticRegression instance is not fitted yet")

		X = np.asarray(X, dtype=float)
		if X.ndim != 2:
			raise ValueError("X must be a 2D array")
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but this model was fitted with {self.n_features_in_}"
			)

		return self._augment_features(X)

	def decision_function(self, X: np.ndarray) -> np.ndarray:
		"""Return class scores before softmax."""
		x_aug = self.prepare_x(X)
		return x_aug @ self.coef_

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		"""Return class probabilities for each row in X."""
		return _softmax(self.decision_function(X))

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""Predict the most likely class label for each row in X."""
		probabilities = self.predict_proba(X)
		class_indices = np.argmax(probabilities, axis=1)
		return self.classes_[class_indices]

	def score(self, X: np.ndarray, y: np.ndarray) -> float:
		"""Return mean classification accuracy."""
		y_true = np.asarray(y)
		y_pred = self.predict(X)
		return float(np.mean(y_pred == y_true))
