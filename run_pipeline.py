import os
import sys
import numpy as np

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from param_estimation.export_manager import SignalExporter
from param_estimation.parameter_extractor import ParameterExtractor
from deinterleave_fec.bitstream_correlation import correlate_and_frame
from deinterleave_fec.interleaver import deinterleave_bits
from deinterleave_fec.fec_decoder import hamming_7_4_decode_bits, bits_to_ascii


def execute_pipeline(
    iq_samples: np.ndarray,
    sample_rate: float,
    demod_bitstream: str,
    sync_pattern: str = "10101011",
    has_interleaving: bool = False,
    has_fec: bool = False,
    output_prefix: str = "sih_run",
) -> dict:
    """
    End-to-End Pipeline runner connecting:
    Parameter Extraction -> Real Bitstream Correlation & Framing ->
    De-interleaving -> FEC Decoding -> Data Export.
    """
    print("[1/4] Extracting Signal Parameters...")
    extractor = ParameterExtractor(iq_samples, sample_rate)
    param_report = extractor.get_full_extraction_report(
        modulation_type="QPSK"
    )

    print("[2/4] Running Real Bitstream Correlation & Framing...")
    # Autonomous sync word correlation: finds message boundary without prior position knowledge!
    frame_report = correlate_and_frame(
        demod_bitstream,
        sync_pattern=sync_pattern,
        header_length_bits=0,
        confidence_threshold=0.75,
        auto_invert=True
    )
    raw_payload = frame_report.get("payload_bits", "")
    print(f"  - Sync Pattern Matched: {frame_report.get('sync_word_matched')}")
    print(f"  - Detected Boundary Index: {frame_report.get('sync_index')}")
    print(f"  - Correlation Score: {frame_report.get('correlation_score', 0):.2f}")

    print("[3/4] Processing De-interleaving & FEC Decoding...")
    processed_bits = raw_payload
    if has_interleaving and processed_bits:
        processed_bits = deinterleave_bits(processed_bits, cols=8)
        print("  - Block De-interleaving applied (cols=8)")

    corrected_errors = 0
    if has_fec and processed_bits:
        decoded_bits, corrected_errors = hamming_7_4_decode_bits(processed_bits, original_length=56)
        print(f"  - Hamming(7,4) FEC decoded (corrected {corrected_errors} bit errors)")
    else:
        decoded_bits = processed_bits

    decoded_text = bits_to_ascii(decoded_bits) if decoded_bits else ""
    print(f"  - Recovered Payload Text: {decoded_text!r}")

    payload_report = {
        "sync_found": frame_report.get("status") == "sync_found",
        "sync_index": frame_report.get("sync_index", -1),
        "correlation_score": frame_report.get("correlation_score", 0.0),
        "is_inverted": frame_report.get("is_inverted", False),
        "raw_payload_bits": raw_payload,
        "decoded_bits": decoded_bits,
        "decoded_text": decoded_text,
        "fec_errors_corrected": corrected_errors,
    }

    # Combine all extracted metrics into a master dictionary
    master_report = {**param_report, **payload_report}

    print("[4/4] Exporting Results...")
    exporter = SignalExporter(output_dir="final_exports")
    export_paths = exporter.export_all(
        master_report,
        prefix=output_prefix,
        payload_bitstring=decoded_bits,
    )

    print("\n[SUCCESS] Pipeline Execution Complete!")
    return export_paths


if __name__ == "__main__":
    # Synthetic Signal Data for Test Run
    fs = 2000000  # 2 MHz sample rate
    t = np.linspace(0, 0.005, int(fs * 0.005))
    synthetic_iq = np.exp(1j * 2 * np.pi * 50000 * t) + 0.05 * (
        np.random.randn(len(t)) + 1j * np.random.randn(len(t))
    )

    # Simulated Demodulated Bitstream: Noise + Sync Word ('10101011') + 'SIH2026'
    mock_sync = "10101011"
    mock_payload = (
        "01010011010010010100100000110010001100000011001000110110"  # 'SIH2026'
    )
    # Prepend 16 random noise bits to demonstrate finding boundary itself
    mock_bitstream = "0000111110100000" + mock_sync + mock_payload

    # Run End-to-End Pipeline
    saved_files = execute_pipeline(
        iq_samples=synthetic_iq,
        sample_rate=fs,
        demod_bitstream=mock_bitstream,
        sync_pattern=mock_sync,
        has_interleaving=False,
        has_fec=False,
        output_prefix="pipeline_test",
    )

    print("\nGenerated Output Files:")
    for key, path in saved_files.items():
        print(f" - {key}: {path}")
