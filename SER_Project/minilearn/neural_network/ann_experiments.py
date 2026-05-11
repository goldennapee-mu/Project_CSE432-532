"""
ANN experiment helpers module.

This implementation lives beside the neural_network implementation
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler

from .ann import SimpleANNClassifier

def load_dataset(data_path: Path) -> tuple:
	frame = pd.read_csv(data_path)
	target_col = "emotion_code" if "emotion_code" in frame.columns else "emotion"
	drop_cols = [
		column_name for column_name in [
			"filepath", "filename", "emotion", "emotion_code", "vocal_channel",
			"modality", "intensity", "statement", "actor_id", "repetition",
		] if column_name in frame.columns
	]
	features = frame.drop(columns=drop_cols)
	labels = frame[target_col]
	return features, labels


def tune_simple_ann(
	data_path: str | Path,
	output_path: str | Path | None = None,
	random_state: int = 42,
) -> dict:
	data_path = Path(data_path)
	output_path = Path(output_path) if output_path is not None else data_path.with_name("ann_tune_results.json")

	X, y = load_dataset(data_path)
	X_train, x_val, y_train, y_val = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=random_state,
		stratify=y,
	)

	scaler = StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	x_val = scaler.transform(x_val)

	grid = ParameterGrid(
		{
			"hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
			"learning_rate": [0.01, 0.001],
			"batch_size": [32, 64],
			"activation": ["relu", "tanh"],
			"l2_penalty": [1e-4, 1e-3],
			"max_epochs": [50, 100],
		}
	)

	results = []
	best_result = {"macro_f1": -1.0, "params": None}

	for params in grid:
		model = SimpleANNClassifier(
			hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
			learning_rate=params["learning_rate"],
			batch_size=params["batch_size"],
			activation=params["activation"],
			l2_penalty=params["l2_penalty"],
			max_epochs=params["max_epochs"],
			random_state=random_state,
		)
		model.fit(X_train, y_train)
		predictions = model.predict(x_val)
		macro_f1 = float(f1_score(y_val, predictions, average="macro"))
		results.append({"params": params, "macro_f1": macro_f1})
		if macro_f1 > best_result["macro_f1"]:
			best_result = {"macro_f1": macro_f1, "params": params}

	payload = {"best": best_result, "results": sorted(results, key=lambda item: item["macro_f1"], reverse=True)}
	output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
	return payload