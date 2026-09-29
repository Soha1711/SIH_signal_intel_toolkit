import csv
from pathlib import Path

from param_estimation.modulation_classifier import ModulationClassifier


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "synthetic_dataset"
    / "manifest.csv"
)

OUTPUT_DIR = ROOT / "final_exports"

RESULTS = []


def load_manifest():
    """Load the synthetic dataset manifest."""

    with open(
        MANIFEST,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def load_iq_file(file_path):
    """
    Load headerless float32 IQ data.

    Dataset format:
        I0, Q0, I1, Q1, ...
    """

    import numpy as np

    raw = np.fromfile(
        file_path,
        dtype=np.float32,
    )

    if len(raw) % 2 != 0:
        raw = raw[:-1]

    iq = (
        raw[0::2]
        + 1j * raw[1::2]
    )

    return iq


def test_modulation_classifier():
    """
    Evaluate the ML modulation classifier
    against the complete synthetic dataset.
    """

    rows = load_manifest()

    classifier = ModulationClassifier()

    correct = 0
    total = 0

    for row in rows:

        file_path = (
            ROOT / row["file_path"]
        )

        expected = row["modulation"]

        signal = load_iq_file(
            file_path
        )

        predicted, confidence = (
            classifier.classify(signal)
        )

        passed = (
            predicted.upper()
            == expected.upper()
        )

        if passed:
            correct += 1

        total += 1

        RESULTS.append(
            {
                "file": row["file_name"],
                "expected_modulation": expected,
                "predicted_modulation": predicted,
                "snr_db": row["snr_db"],
                "confidence": confidence,
                "correct": passed,
            }
        )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    print(
        f"\nClassifier accuracy: "
        f"{accuracy * 100:.2f}% "
        f"({correct}/{total})"
    )

    # The initial goal is to measure the classifier.
    # Do not impose an arbitrary accuracy threshold yet.
    assert total == 96


def test_write_classification_results():
    """
    Export classifier evaluation results
    to CSV.
    """

    if not RESULTS:
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / "classification_results.csv"
    )

    fieldnames = [
        "file",
        "expected_modulation",
        "predicted_modulation",
        "snr_db",
        "confidence",
        "correct",
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(RESULTS)

    print(
        f"\nClassification results written to: "
        f"{output_file}"
    )