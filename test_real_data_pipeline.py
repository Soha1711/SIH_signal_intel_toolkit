import os
import numpy as np
from param_estimation.export_manager import SignalExporter
from param_estimation.parameter_extractor import ParameterExtractor


def test_on_real_file(file_path: str, sample_rate: float = 2000000.0):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📂 Reading real dataset file: {file_path}")

    # Read raw interleaved IQ data (I, Q, I, Q...) and convert to complex array
    try:
        raw_data = np.fromfile(file_path, dtype=np.int16)
        # Convert to float & form complex samples
        i_samples = raw_data[0::2].astype(np.float32)
        q_samples = raw_data[1::2].astype(np.float32)
        samples = i_samples + 1j * q_samples
    except Exception as e:
        print(f"❌ Error reading raw IQ file: {e}")
        return

    print(
        f"📊 Loaded {len(samples)} complex samples. Sample Rate: {sample_rate}"
        " Hz"
    )

    # Step 2: Extract RF Parameters
    extractor = ParameterExtractor(samples, sample_rate=sample_rate)
    extracted_metrics = extractor.get_full_extraction_report(
        modulation_type="Estimated_RF"
    )

    print("\n🔍 Extracted RF Metrics:")
    for key, val in extracted_metrics.items():
        print(f"  - {key}: {val}")

    # Step 3: Export Results
    exporter = SignalExporter(output_dir="real_data_exports")
    export_paths = exporter.export_all(
        extracted_metrics, prefix="real_file_test"
    )

    print("\n📁 Export Output Success:")
    for fmt, path in export_paths.items():
        print(f"  - {fmt}: {path}")


if __name__ == "__main__":
    sample_file = os.path.join("sample_data", "test_tone.iq")
    test_on_real_file(sample_file)