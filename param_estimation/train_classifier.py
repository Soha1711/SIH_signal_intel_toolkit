"""
SIH 2026 - Signal Intelligence Toolkit
ML Modulation Classifier (Random Forest)
Extracts statistical/cumulant features from IQ data and trains a classifier.
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def extract_signal_features(iq_signal: np.ndarray) -> np.array:
    """
    Extracts statistical & higher-order cumulant features from IQ signal.
    """
    # Normalize signal power
    iq_signal = iq_signal / np.sqrt(np.mean(np.abs(iq_signal) ** 2) + 1e-12)

    amp = np.abs(iq_signal)
    phase = np.angle(iq_signal)

    # 1. Basic Statistical Moments
    mean_amp = np.mean(amp)
    std_amp = np.std(amp)
    var_amp = np.var(amp)
    std_phase = np.std(phase)

    # 2. Moments & Cumulants
    m20 = np.mean(iq_signal ** 2)
    m21 = np.mean(np.abs(iq_signal) ** 2)
    m40 = np.mean(iq_signal ** 4)
    m42 = np.mean((np.abs(iq_signal) ** 2) * (iq_signal ** 2))

    # Higher-Order Cumulants
    c40 = m40 - 3 * (m20 ** 2)
    c42 = m42 - np.abs(m20) ** 2 - 2 * (m21 ** 2)

    abs_c40 = np.abs(c40)
    abs_c42 = np.abs(c42)

    # 3. Spectral Features
    fft_vals = np.abs(np.fft.fft(iq_signal))
    spectral_std = np.std(fft_vals)
    spectral_max = np.max(fft_vals)

    return np.array([
        mean_amp, std_amp, var_amp, std_phase,
        np.abs(m20), m21, abs_c40, abs_c42,
        spectral_std, spectral_max
    ])


def train_modulation_classifier(dataset_dir: str = "synthetic_dataset"):
    ds_path = Path(dataset_dir)
    manifest_path = ds_path / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    X = []
    y = []

    print("📊 Extracting features from synthetic dataset...")
    for entry in manifest:
        file_path = entry["file_path"]
        mod_type = entry["modulation"]

        # Read IQ signal
        iq_data = np.fromfile(file_path, dtype=np.complex64)
        if len(iq_data) == 0:
            continue

        features = extract_signal_features(iq_data)
        X.append(features)
        y.append(mod_type)

    X = np.array(X)
    y = np.array(y)

    print(f"✅ Features extracted for {len(X)} samples.")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Evaluation
    y_pred = rf_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model Training Complete! Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    # Save Model
    model_path = Path("param_estimation") / "modulation_classifier.pkl"
    joblib.dump(rf_clf, model_path)
    print(f"💾 Trained Model saved to: {model_path.resolve()}")


if __name__ == "__main__":
    train_modulation_classifier()