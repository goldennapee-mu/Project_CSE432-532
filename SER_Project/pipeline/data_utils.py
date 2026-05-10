"""
Utilities for parsing and building RAVDESS metadata.

Contains functions to parse RAVDESS audio filenames and build structured
metadata tables from a collection of audio files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import pandas as pd

PathLike = Union[str, Path]

RAVDESS_MODALITY = {
    1: "full-AV",
    2: "video-only",
    3: "audio-only",
}

RAVDESS_VOCAL_CHANNEL = {
    1: "speech",
    2: "song",
}

RAVDESS_EMOTION = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

RAVDESS_INTENSITY = {
    1: "normal",
    2: "strong",
}

RAVDESS_STATEMENT = {
    1: "Kids are talking by the door",
    2: "Dogs are sitting by the door",
}

def parse_ravdess_filename(filepath: PathLike) -> dict:
    """
    Parse one RAVDESS filename into codes and human-readable labels.

    Expected filename format (without extension):
        MM-VC-EE-II-SS-RR-AA

    Example:
        03-01-05-01-02-01-12.wav

    Args:
        filepath: Path to a RAVDESS audio file, or a raw filename.

    Returns:
        A dictionary containing both numeric code fields and decoded labels.

    Raises:
        ValueError: If the filename does not have exactly 7 dash-separated
            fields, or if any field is non-numeric.
    """
    path = Path(filepath)
    # Split filename stem by dashes to get the 7-part identifier
    parts = path.stem.split("-")

    if len(parts) != 7:
        raise ValueError(f"Invalid RAVDESS filename format: {path.name}")

    try:
        modality, vocal_channel, emotion, intensity, statement, repetition, actor = map(int, parts)
    except ValueError as exc:
        raise ValueError(f"Filename contains non-numeric fields: {path.name}") from exc

    # Return both codes and decoded labels for flexibility
    return {
        "filepath": str(path),
        "filename": path.name,
        "modality_code": modality,
        "vocal_channel_code": vocal_channel,
        "emotion_code": emotion,
        "intensity_code": intensity,
        "statement_code": statement,
        "repetition": repetition,
        "actor_id": actor,
        "modality": RAVDESS_MODALITY.get(modality, "unknown"),
        "vocal_channel": RAVDESS_VOCAL_CHANNEL.get(vocal_channel, "unknown"),
        "emotion": RAVDESS_EMOTION.get(emotion, "unknown"),
        "intensity": RAVDESS_INTENSITY.get(intensity, "unknown"),
        "statement": RAVDESS_STATEMENT.get(statement, "unknown"),
    }

def build_ravdess_metadata(filepaths: Iterable[PathLike]) -> pd.DataFrame:
    """
    Build a metadata table by parsing a collection of RAVDESS files.

    Args:
        filepaths: Iterable of file paths or filenames.

    Returns:
        A pandas DataFrame with one row per file and parsed metadata columns.

    Raises:
        ValueError: Propagated from ``parse_ravdess_filename`` if any filename
            is malformed.
    """
    rows = [parse_ravdess_filename(fp) for fp in filepaths]
    return pd.DataFrame(rows)
