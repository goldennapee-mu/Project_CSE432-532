"""
Audio feature extraction pipeline for the RAVDESS SER project.

This script scans a data directory for WAV files, parses each filename using
`pipeline.data_utils.parse_ravdess_filename`, extracts frame-level acoustic
features with librosa, summarizes each feature over time, and exports one row
per file to a CSV file.

The output table is designed to be consumed by later notebooks and model
training scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .data_utils import parse_ravdess_filename
except ImportError:
    # Supports direct script execution: python pipeline/extract_features.py
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.data_utils import parse_ravdess_filename

# Default sample rate for loading audio. RAVDESS is distributed at 48 kHz.
DEFAULT_SAMPLE_RATE = 48_000

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract audio features from RAVDESS WAV files and save CSV.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Path to directory containing RAVDESS audio files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features.csv"),
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Target sample rate used when loading audio.",
    )
    parser.add_argument(
        "--n_mfcc",
        type=int,
        default=13,
        help="Number of MFCC coefficients to compute.",
    )
    parser.add_argument(
        "--include_song",
        action="store_true",
        help="Include song files in addition to speech files.",
    )
    parser.add_argument(
        "--stats",
        nargs="+",
        default=["mean", "std"],
        choices=["mean", "std", "min", "max", "median"],
        help="Summary statistics to compute over time for each feature.",
    )
    return parser.parse_args()

def discover_audio_files(data_dir: Path) -> list[Path]:
    """Recursively discover WAV files under a data directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    return sorted(data_dir.rglob("*.wav"))

def _summarize_vector(values: np.ndarray, prefix: str, stats: list[str]) -> dict[str, float]:
    """Summarize a 1D vector with selected statistics.

    Args:
        values: One-dimensional array of frame-level values.
        prefix: Prefix used to name output keys.
        stats: List of statistic names to compute.

    Returns:
        Dictionary mapping feature names to numeric summaries.
    """
    summary: dict[str, float] = {}

    if "mean" in stats:
        summary[f"{prefix}_mean"] = float(np.mean(values))
    if "std" in stats:
        summary[f"{prefix}_std"] = float(np.std(values))
    if "min" in stats:
        summary[f"{prefix}_min"] = float(np.min(values))
    if "max" in stats:
        summary[f"{prefix}_max"] = float(np.max(values))
    if "median" in stats:
        summary[f"{prefix}_median"] = float(np.median(values))

    return summary

def _summarize_matrix(values: np.ndarray, prefix: str, stats: list[str]) -> dict[str, float]:
    """Summarize a 2D feature matrix (coefficients x frames).

    For each coefficient row, this function computes selected summary stats
    over time and returns flattened names like `mfcc_00_mean`.
    """
    summary: dict[str, float] = {}

    for idx, row in enumerate(values):
        coeff_prefix = f"{prefix}_{idx:02d}"
        summary.update(_summarize_vector(row, coeff_prefix, stats))

    return summary

def extract_features_from_audio(
    y: np.ndarray,
    sr: int,
    n_mfcc: int,
    stats: list[str],
) -> dict[str, float]:
    """Extract and summarize acoustic features from one audio waveform."""
    features: dict[str, float] = {}

    # MFCC and dynamic derivatives capture timbral structure and change.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features.update(_summarize_matrix(mfcc, "mfcc", stats))
    features.update(_summarize_matrix(mfcc_delta, "mfcc_delta", stats))
    features.update(_summarize_matrix(mfcc_delta2, "mfcc_delta2", stats))

    # Chroma provides coarse pitch-class energy and is often useful for song.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.update(_summarize_matrix(chroma, "chroma", stats))

    # Mel-spectrogram summary provides broader spectral envelope information.
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.update(_summarize_matrix(mel_db, "mel", stats))

    # One-dimensional frame-level features.
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    features.update(_summarize_vector(zcr, "zcr", stats))
    features.update(_summarize_vector(rms, "rms", stats))
    features.update(_summarize_vector(centroid, "spectral_centroid", stats))
    features.update(_summarize_vector(bandwidth, "spectral_bandwidth", stats))
    features.update(_summarize_vector(rolloff, "spectral_rolloff", stats))

    # A few global audio properties are useful for debugging and analysis.
    features["duration_seconds"] = float(librosa.get_duration(y=y, sr=sr))
    features["signal_abs_mean"] = float(np.mean(np.abs(y)))

    return features

def extract_features_for_file(
    audio_path: Path,
    sr: int,
    n_mfcc: int,
    stats: list[str],
) -> dict[str, Any]:
    """Parse metadata and extract acoustic features for a single WAV file."""
    metadata = parse_ravdess_filename(audio_path)

    # Force mono for consistent feature shapes across files.
    y, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
    loaded_sr = int(loaded_sr)
    feature_values = extract_features_from_audio(y=y, sr=loaded_sr, n_mfcc=n_mfcc, stats=stats)

    # Merge metadata and numeric feature summaries into one record.
    return {**metadata, **feature_values}

def build_feature_dataframe(
    filepaths: list[Path],
    sr: int,
    n_mfcc: int,
    stats: list[str],
    include_song: bool,
) -> pd.DataFrame:
    """Build a feature table from a list of audio files.

    Files that fail parsing or feature extraction are skipped and reported.
    """
    rows: list[dict[str, Any]] = []
    skipped = 0

    for path in tqdm(filepaths, desc="Extracting features", unit="file"):
        try:
            row = extract_features_for_file(path, sr=sr, n_mfcc=n_mfcc, stats=stats)

            if not include_song and row.get("vocal_channel") == "song":
                continue

            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"[WARN] Skipping {path.name}: {exc}")

    if not rows:
        raise RuntimeError("No features were extracted. Check inputs and flags.")

    df = pd.DataFrame(rows)
    df = df.sort_values("filename").reset_index(drop=True)

    if skipped > 0:
        print(f"[INFO] Skipped files: {skipped}")

    return df

def main() -> None:
    """Run the end-to-end feature extraction pipeline from CLI arguments."""
    args = parse_args()

    audio_files = discover_audio_files(args.data_dir)
    print(f"[INFO] Found {len(audio_files)} .wav files under {args.data_dir}")

    features_df = build_feature_dataframe(
        filepaths=audio_files,
        sr=args.sr,
        n_mfcc=args.n_mfcc,
        stats=args.stats,
        include_song=args.include_song,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.output, index=False)

    print(f"[INFO] Feature table shape: {features_df.shape}")
    print(f"[INFO] Saved feature CSV to: {args.output}")

    if "emotion" in features_df.columns:
        print("[INFO] Emotion counts:")
        print(features_df["emotion"].value_counts().sort_index())

if __name__ == "__main__":
    main()
