"""
ANN neural_network module.

This implementation serves as a very simple Neural Network
for use in MiniLearn notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
	shifted = logits - np.max(logits, axis=1, keepdims=True)
	exponentiated = np.exp(shifted)
	return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


@dataclass
class SimpleANNClassifier:
	hidden_layer_sizes: int | tuple[int, ...] = (64,)
	activation: str = "relu"
	learning_rate: float = 0.01
	max_epochs: int = 200
	batch_size: int | None = None
	l2_penalty: float = 0.0
	tol: float = 1e-4
	random_state: int | None = None
	verbose: bool = False

	def __post_init__(self) -> None:
		if isinstance(self.hidden_layer_sizes, int):
			self.hidden_layer_sizes = (self.hidden_layer_sizes,)
		else:
			self.hidden_layer_sizes = tuple(self.hidden_layer_sizes)
		if len(self.hidden_layer_sizes) == 0:
			raise ValueError("hidden_layer_sizes must contain at least one layer")
		if any(size <= 0 for size in self.hidden_layer_sizes):
			raise ValueError("hidden_layer_sizes must contain only positive integers")
		if self.learning_rate <= 0:
			raise ValueError("learning_rate must be positive")
		if self.max_epochs <= 0:
			raise ValueError("max_epochs must be positive")
		if self.batch_size is not None and self.batch_size <= 0:
			raise ValueError("batch_size must be positive when provided")
		if self.l2_penalty < 0:
			raise ValueError("l2_penalty cannot be negative")
		if self.activation not in {"relu", "tanh", "sigmoid"}:
			raise ValueError("activation must be one of: relu, tanh, sigmoid")

	def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		X = np.asarray(X, dtype=float)
		y = np.asarray(y)
		if X.ndim != 2:
			raise ValueError("X must be a 2D array")
		if y.ndim != 1:
			raise ValueError("y must be a 1D array")
		if X.shape[0] != y.shape[0]:
			raise ValueError("X and y must contain the same number of rows")
		return X, y

	def _activation_forward(self, Z: np.ndarray) -> np.ndarray:
		if self.activation == "relu":
			return np.maximum(0.0, Z)
		if self.activation == "tanh":
			return np.tanh(Z)
		return 1.0 / (1.0 + np.exp(-Z))

	def _activation_backward(self, Z: np.ndarray) -> np.ndarray:
		if self.activation == "relu":
			return (Z > 0).astype(float)
		if self.activation == "tanh":
			activated = np.tanh(Z)
			return 1.0 - activated * activated
		activated = 1.0 / (1.0 + np.exp(-Z))
		return activated * (1.0 - activated)

	def _encode_labels(self, y: np.ndarray) -> np.ndarray:
		self.classes_ = np.unique(y)
		self.n_classes_ = self.classes_.shape[0]
		class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
		return np.array([class_to_index[label] for label in y], dtype=int)

	def _initialize_parameters(self, n_features: int, rng: np.random.Generator) -> None:
		layer_sizes = [n_features, *self.hidden_layer_sizes, self.n_classes_]
		self.weights_ = []
		self.biases_ = []
		for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
			scale = np.sqrt(2.0 / input_size) if self.activation == "relu" else np.sqrt(1.0 / input_size)
			self.weights_.append(rng.normal(0.0, scale, size=(input_size, output_size)))
			self.biases_.append(np.zeros(output_size, dtype=float))

	def _forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
		activations = [X]
		pre_activations: list[np.ndarray] = []
		current = X
		for weight_matrix, bias_vector in zip(self.weights_[:-1], self.biases_[:-1]):
			Z = current @ weight_matrix + bias_vector
			current = self._activation_forward(Z)
			pre_activations.append(Z)
			activations.append(current)
		logits = current @ self.weights_[-1] + self.biases_[-1]
		return logits, activations, pre_activations

	def _compute_loss(self, logits: np.ndarray, y_one_hot: np.ndarray) -> float:
		probabilities = _softmax(logits)
		loss = -np.mean(np.sum(y_one_hot * np.log(probabilities + 1e-12), axis=1))
		if self.l2_penalty > 0:
			loss += 0.5 * self.l2_penalty * sum(np.sum(weight_matrix * weight_matrix) for weight_matrix in self.weights_)
		return loss

	def _backward(
		self,
		activations: list[np.ndarray],
		pre_activations: list[np.ndarray],
		probabilities: np.ndarray,
		y_one_hot: np.ndarray,
	) -> tuple[list[np.ndarray], list[np.ndarray]]:
		batch_size = y_one_hot.shape[0]
		delta = (probabilities - y_one_hot) / batch_size
		gradients_w = [np.zeros_like(weight_matrix) for weight_matrix in self.weights_]
		gradients_b = [np.zeros_like(bias_vector) for bias_vector in self.biases_]
		gradients_w[-1] = activations[-1].T @ delta
		gradients_b[-1] = np.sum(delta, axis=0)
		if self.l2_penalty > 0:
			gradients_w[-1] += self.l2_penalty * self.weights_[-1]
		for layer_index in range(len(self.weights_) - 2, -1, -1):
			delta = (delta @ self.weights_[layer_index + 1].T) * self._activation_backward(pre_activations[layer_index])
			gradients_w[layer_index] = activations[layer_index].T @ delta
			gradients_b[layer_index] = np.sum(delta, axis=0)
			if self.l2_penalty > 0:
				gradients_w[layer_index] += self.l2_penalty * self.weights_[layer_index]
		return gradients_w, gradients_b

	def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleANNClassifier":
		X, y = self._validate_training_data(X, y)
		y_indices = self._encode_labels(y)
		self.n_features_in_ = X.shape[1]
		rng = np.random.default_rng(self.random_state)
		self._initialize_parameters(self.n_features_in_, rng)
		y_one_hot = np.eye(self.n_classes_, dtype=float)[y_indices]
		batch_size = self.batch_size or X.shape[0]
		previous_loss = np.inf
		for epoch in range(self.max_epochs):
			indices = rng.permutation(X.shape[0])
			X_epoch = X[indices]
			y_epoch = y_one_hot[indices]
			for start in range(0, X.shape[0], batch_size):
				stop = start + batch_size
				logits, activations, pre_activations = self._forward(X_epoch[start:stop])
				probabilities = _softmax(logits)
				gradients_w, gradients_b = self._backward(activations, pre_activations, probabilities, y_epoch[start:stop])
				for layer_index in range(len(self.weights_)):
					self.weights_[layer_index] -= self.learning_rate * gradients_w[layer_index]
					self.biases_[layer_index] -= self.learning_rate * gradients_b[layer_index]
			loss = self._compute_loss(self._forward(X)[0], y_one_hot)
			if self.verbose and epoch % 10 == 0:
				print(f"epoch={epoch} loss={loss:.6f}")
			if abs(previous_loss - loss) < self.tol:
				break
			previous_loss = loss
		return self

	def prepare_x(self, X: np.ndarray) -> np.ndarray:
		if not hasattr(self, "weights_"):
			raise ValueError("This SimpleANNClassifier instance is not fitted yet")
		X = np.asarray(X, dtype=float)
		if X.ndim != 2:
			raise ValueError("X must be a 2D array")
		if X.shape[1] != self.n_features_in_:
			raise ValueError(f"X has {X.shape[1]} features, but this model was fitted with {self.n_features_in_}")
		return X

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		X = self.prepare_x(X)
		return _softmax(self._forward(X)[0])

	def predict(self, X: np.ndarray) -> np.ndarray:
		class_indices = np.argmax(self.predict_proba(X), axis=1)
		return self.classes_[class_indices]

	def score(self, X: np.ndarray, y: np.ndarray) -> float:
		return float(np.mean(self.predict(X) == np.asarray(y)))