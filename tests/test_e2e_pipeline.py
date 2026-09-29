import csv
from pathlib import Path

import pytest

from pipeline.orchestrator import run_full_pipeline


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "synthetic_dataset"
    / "manifest.csv"
)


# ============================================================
# E2E RESULTS
# ============================================================

RESULTS = []
@pytest.fixture(scope="session", autouse=True)
def write_e2e_results():
    """
    Write E2E results after all tests finish.
    """
    yield

    if not RESULTS:
        print("\nNo E2E results were collected.")
        return

    output_dir = ROOT / "final_exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir / "e2e_results.csv"
    )

    fieldnames = [
        "file",
        "modulation",
        "snr_db",
        "fec",
        "interleaver",
        "detected_modulation",
        "confidence",
        "errors_corrected",
        "decoded_text",
        "passed",
        "error",
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
        f"\nWrote E2E results to: {output_file}"
    )


# ============================================================
# MANIFEST LOADING
# ============================================================

def load_manifest():
    """
    Load the synthetic dataset manifest.
    """

    with open(
        MANIFEST,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        return list(
            csv.DictReader(f)
        )


# ============================================================
# SELECT E2E CASES
# ============================================================

def select_cases():
    """
    Select representative E2E cases.

    2 modulations:
        BPSK
        QPSK

    3 SNR levels:
        30 dB
        10 dB
        5 dB

    2 FEC states:
        OFF
        ON

    2 interleaving states:
        OFF
        ON

    Total:
        2 × 3 × 2 × 2 = 24 cases
    """

    rows = load_manifest()

    selected = []

    for modulation in [
        "BPSK",
        "QPSK",
    ]:

        for snr in [
            30,
            10,
            5,
        ]:

            for fec in [
                "False",
                "True",
            ]:

                for interleave in [
                    "False",
                    "True",
                ]:

                    matches = [
                        row
                        for row in rows
                        if (
                            row["modulation"]
                            == modulation
                            and int(row["snr_db"])
                            == snr
                            and row["has_fec"]
                            == fec
                            and row["has_interleaving"]
                            == interleave
                        )
                    ]

                    if not matches:
                        raise AssertionError(
                            "No dataset case found for "
                            f"{modulation}, "
                            f"{snr} dB, "
                            f"FEC={fec}, "
                            f"interleave={interleave}"
                        )

                    selected.append(
                        matches[0]
                    )

    return selected


# ============================================================
# SELECT CASES
# ============================================================

CASES = select_cases()


# ============================================================
# MAIN E2E TEST
# ============================================================

@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda row: row["file_name"],
)
def test_e2e_pipeline(case):
    """
    Run the complete pipeline against
    a synthetic dataset case.
    """

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    file_path = (
        ROOT / case["file_path"]
    )

    expected_modulation = (
        case["modulation"]
    )

    expected_text = (
        case["raw_payload_text"]
    )

    has_fec = (
        case["has_fec"] == "True"
    )

    has_interleaving = (
        case["has_interleaving"]
        == "True"
    )

    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    result = run_full_pipeline(
        file_path=str(file_path),

        sample_rate=float(
            case["sample_rate"]
        ),

        modulation_override=(
            expected_modulation
        ),

        expected_text=expected_text,

        has_fec=has_fec,

        has_interleaving=(
            has_interleaving
        ),

        fec_scheme="hamming",

        interleaver_type="block",
    )

    # --------------------------------------------------------
    # Determine whether this case passed
    # --------------------------------------------------------

    case_passed = (
        result.get("error") is None
        and result.get(
            "ascii_text",
            "",
        ) == expected_text
        and result.get(
            "matches_expected"
        ) is True
    )

    # --------------------------------------------------------
    # Store result for CSV
    # --------------------------------------------------------

    RESULTS.append(
        {
            "file": case["file_name"],

            "modulation": case.get(
                "modulation",
                "",
            ),

            "snr_db": case.get(
                "snr_db",
                "",
            ),

            "fec": case.get(
                "has_fec",
                "",
            ),

            "interleaver": case.get(
                "has_interleaving",
                "",
            ),

            "detected_modulation": (
                result.get(
                    "modulation_type",
                    "",
                )
            ),

            "confidence": (
                result.get(
                    "confidence",
                    "",
                )
            ),

            "errors_corrected": (
                result.get(
                    "error_count",
                    0,
                )
            ),

            "decoded_text": (
                result.get(
                    "ascii_text",
                    "",
                )
            ),

            "passed": case_passed,

            "error": (
                result.get(
                    "error",
                    "",
                )
                or ""
            ),
        }
    )

    # ========================================================
    # ASSERTIONS
    # ========================================================

    # Pipeline must not report an error.
    assert result.get("error") is None, (
        f"{case['file_name']} failed:\n"
        f"{result.get('error')}"
    )

    # Required stages must have completed.
    assert len(
        result.get(
            "stage_log",
            [],
        )
    ) >= 8

    # Modulation must match manifest.
    assert (
        result.get(
            "modulation_type"
        )
        == expected_modulation
    )

    # Demodulation must produce bits.
    bitstream = result.get(
        "bitstream",
        "",
    )

    assert bitstream

    assert all(
        bit in "01"
        for bit in str(bitstream)
    )

    # Synchronization should be detected.
    expected_sync = (
        case["sync_word"]
    )

    assert (
        result.get(
            "sync_word"
        )
        == expected_sync
    )

    # Final ASCII payload.
    ascii_text = result.get(
        "ascii_text",
        "",
    )

    assert ascii_text == expected_text, (
        f"{case['file_name']} "
        f"decoded incorrect payload.\n"
        f"Expected: {expected_text!r}\n"
        f"Got:      {ascii_text!r}\n"
        f"FEC: {has_fec}, "
        f"Interleave: "
        f"{has_interleaving}, "
        f"SNR: {case['snr_db']} dB"
    )

    # Printable ratio.
    printable_ratio = float(
        result.get(
            "printable_ratio",
            0.0,
        )
    )

    assert printable_ratio >= 0.90, (
        f"{case['file_name']} "
        f"printable ratio too low: "
        f"{printable_ratio}"
    )

    # Expected text validation.
    assert (
        result.get(
            "matches_expected"
        )
        is True
    )


# ============================================================
# MANIFEST TESTS
# ============================================================

def test_manifest_exists():
    """
    Ensure the synthetic dataset manifest
    is available.
    """

    assert MANIFEST.exists(), (
        f"Dataset manifest not found: "
        f"{MANIFEST}"
    )


def test_manifest_has_expected_size():
    """
    The generated dataset should contain
    96 signal entries.
    """

    rows = load_manifest()

    assert len(rows) == 96


def test_manifest_covers_expected_modulations():
    """
    Verify the four generated modulation
    families.
    """

    rows = load_manifest()

    modulations = {
        row["modulation"]
        for row in rows
    }

    assert modulations == {
        "BPSK",
        "QPSK",
        "8PSK",
        "16QAM",
    }