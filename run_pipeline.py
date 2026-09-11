import numpy as np
from param_estimation.export_manager import SignalExporter
from param_estimation.parameter_extractor import ParameterExtractor
from param_estimation.demod_payload import PayloadProcessor


def execute_pipeline(
    iq_samples: np.ndarray,
    sample_rate: float,
    demod_bitstream: str,
    sync_pattern: str = "10101011",
    output_prefix: str = "sih_run",
) -> dict:
    """End-to-End Pipeline runner connecting Parameter Extraction, Frame Sync,

    and Data Export modules.
    """
    print("[1/3] Extracting Signal Parameters...")
    extractor = ParameterExtractor(iq_samples, sample_rate)
    param_report = extractor.get_full_extraction_report(
        modulation_type="QPSK"
    )

    print("[2/3] Processing Bitstream & Payload...")
    processor = PayloadProcessor(demod_bitstream)
    payload_report = processor.extract_full_frame(sync_pattern=sync_pattern)

    # Combine all extracted metrics into a master dictionary
    master_report = {**param_report, **payload_report}

    print("[3/3] Exporting Results...")
    exporter = SignalExporter(output_dir="final_exports")
    export_paths = exporter.export_all(
        master_report,
        prefix=output_prefix,
        payload_bitstring=payload_report["raw_payload_bits"],
    )

    print("\n✅ Pipeline Execution Complete!")
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
    mock_bitstream = "00001111" + mock_sync + mock_payload

    # Run End-to-End Pipeline
    saved_files = execute_pipeline(
        iq_samples=synthetic_iq,
        sample_rate=fs,
        demod_bitstream=mock_bitstream,
        sync_pattern=mock_sync,
        output_prefix="pipeline_test",
    )

    print("\nGenerated Output Files:")
    for key, path in saved_files.items():
        print(f" - {key}: {path}")