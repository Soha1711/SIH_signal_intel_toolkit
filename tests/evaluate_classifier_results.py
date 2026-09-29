import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "final_exports"
    / "classification_results.csv"
)

OUTPUT_DIR = (
    ROOT
    / "final_exports"
)

CONFUSION_FILE = (
    OUTPUT_DIR
    / "classification_confusion_matrix.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "classification_summary.json"
)


MODULATIONS = [
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
]

SNR_LEVELS = [
    "0",
    "5",
    "10",
    "15",
    "20",
    "30",
]


def load_results():
    """Load classifier evaluation results."""

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def calculate_accuracy(rows):
    """Calculate accuracy from result rows."""

    if not rows:
        return 0.0

    correct = sum(
        row["correct"].lower()
        == "true"
        for row in rows
    )

    return correct / len(rows)


def main():

    rows = load_results()

    if not rows:
        raise RuntimeError(
            "classification_results.csv is empty."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    total = len(rows)

    correct = sum(
        row["correct"].lower()
        == "true"
        for row in rows
    )

    incorrect = total - correct

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    low_confidence = [
        row
        for row in rows
        if float(row["confidence"]) < 0.60
    ]

    incorrect_rows = [
        row
        for row in rows
        if row["correct"].lower()
        != "true"
    ]

    # ========================================================
    # AVERAGE CONFIDENCE
    # ========================================================

    average_confidence = (
        sum(
            float(row["confidence"])
            for row in rows
        )
        / total
    )

    # ========================================================
    # PER-MODULATION
    # ========================================================

    modulation_summary = {}

    for modulation in MODULATIONS:

        subset = [
            row
            for row in rows
            if row["expected_modulation"]
            == modulation
        ]

        modulation_summary[modulation] = {
            "total": len(subset),
            "correct": sum(
                row["correct"].lower()
                == "true"
                for row in subset
            ),
            "accuracy": (
                calculate_accuracy(subset)
                if subset
                else 0.0
            ),
            "average_confidence": (
                sum(
                    float(row["confidence"])
                    for row in subset
                )
                / len(subset)
                if subset
                else 0.0
            ),
        }

    # ========================================================
    # PER-SNR
    # ========================================================

    snr_summary = {}

    for snr in SNR_LEVELS:

        subset = [
            row
            for row in rows
            if row["snr_db"] == snr
        ]

        snr_summary[snr] = {
            "total": len(subset),
            "correct": sum(
                row["correct"].lower()
                == "true"
                for row in subset
            ),
            "accuracy": (
                calculate_accuracy(subset)
                if subset
                else 0.0
            ),
            "average_confidence": (
                sum(
                    float(row["confidence"])
                    for row in subset
                )
                / len(subset)
                if subset
                else 0.0
            ),
        }

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    confusion = defaultdict(
        lambda: defaultdict(int)
    )

    for row in rows:

        expected = row[
            "expected_modulation"
        ]

        predicted = row[
            "predicted_modulation"
        ]

        confusion[
            expected
        ][predicted] += 1

    with CONFUSION_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "expected\\predicted"
            ]
            + MODULATIONS
        )

        for expected in MODULATIONS:

            writer.writerow(
                [
                    expected
                ]
                + [
                    confusion[
                        expected
                    ][predicted]
                    for predicted
                    in MODULATIONS
                ]
            )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {
        "dataset": {
            "total_samples": total,
            "modulations": MODULATIONS,
            "snr_levels_db": [
                int(x)
                for x in SNR_LEVELS
            ],
        },

        "overall": {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy,
            "accuracy_percent": (
                accuracy * 100
            ),
            "average_confidence": (
                average_confidence
            ),
            "low_confidence_count": (
                len(low_confidence)
            ),
            "low_confidence_threshold": 0.60,
        },

        "per_modulation": (
            modulation_summary
        ),

        "per_snr": snr_summary,

        "misclassifications": [
            {
                "file": row["file"],
                "expected": row[
                    "expected_modulation"
                ],
                "predicted": row[
                    "predicted_modulation"
                ],
                "snr_db": row["snr_db"],
                "confidence": float(
                    row["confidence"]
                ),
            }
            for row in incorrect_rows
        ],
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # CONSOLE REPORT
    # ========================================================

    print()
    print("=" * 60)
    print("MODULATION CLASSIFIER EVALUATION")
    print("=" * 60)

    print()
    print(
        f"Total samples     : {total}"
    )

    print(
        f"Correct predictions: {correct}"
    )

    print(
        f"Incorrect          : {incorrect}"
    )

    print(
        f"Accuracy           : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Average confidence: "
        f"{average_confidence:.3f}"
    )

    print(
        f"Low confidence     : "
        f"{len(low_confidence)}"
    )

    print()
    print("-" * 60)
    print("PER-MODULATION")
    print("-" * 60)

    for modulation in MODULATIONS:

        data = modulation_summary[
            modulation
        ]

        print(
            f"{modulation:6} : "
            f"{data['correct']}/"
            f"{data['total']} "
            f"({data['accuracy'] * 100:.2f}%)"
        )

    print()
    print("-" * 60)
    print("PER-SNR")
    print("-" * 60)

    for snr in SNR_LEVELS:

        data = snr_summary[snr]

        print(
            f"{snr:>2} dB : "
            f"{data['correct']}/"
            f"{data['total']} "
            f"({data['accuracy'] * 100:.2f}%)"
        )

    print()
    print("-" * 60)
    print("MISCLASSIFICATIONS")
    print("-" * 60)

    if incorrect_rows:

        for row in incorrect_rows:

            print(
                f"{row['file']}: "
                f"{row['expected_modulation']} "
                f"-> "
                f"{row['predicted_modulation']} "
                f"(confidence="
                f"{row['confidence']})"
            )

    else:
        print("None")

    print()
    print("-" * 60)

    print(
        f"Saved confusion matrix: "
        f"{CONFUSION_FILE}"
    )

    print(
        f"Saved summary: "
        f"{SUMMARY_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()