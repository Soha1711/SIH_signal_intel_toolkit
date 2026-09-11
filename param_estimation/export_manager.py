import csv
import json
import os
from typing import Any, Dict


class SignalExporter:
    """Handles formatted exports of extracted RF parameters, frame analysis,

    and demodulated payload bits into JSON, CSV, and raw binary formats.
    """

    def __init__(self, output_dir: str = "exports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_json(
        self, data: Dict[str, Any], filename: str = "signal_summary.json"
    ) -> str:
        """Exports full metadata and parameter extraction results as a JSON file."""
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return file_path

    def export_csv(
        self, data: Dict[str, Any], filename: str = "signal_parameters.csv"
    ) -> str:
        """Exports extracted scalar metrics and signal parameters as a CSV key-value table."""
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter Name", "Extracted Value"])

            for key, val in data.items():
                # Skip complex nested dictionaries/lists for flat CSV summary
                if not isinstance(val, (dict, list)):
                    writer.writerow([key, val])
        return file_path

    def export_payload_binary(
        self,
        bitstring: str,
        filename: str = "extracted_payload.bin",
        as_text: bool = False,
    ) -> str:
        """Saves recovered bitstream/payload to file.

        Optionally converts binary string to ASCII text if readable.
        """
        file_path = os.path.join(self.output_dir, filename)

        if as_text:
            # Convert bitstring (e.g. '01000001') to ASCII bytes
            bytes_list = []
            for i in range(0, len(bitstring) - len(bitstring) % 8, 8):
                byte_chunk = bitstring[i : i + 8]
                bytes_list.append(int(byte_chunk, 2))
            with open(file_path, "wb") as f:
                f.write(bytes(bytes_list))
        else:
            # Save raw '0' and '1' string
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(bitstring)

        return file_path

    def export_all(
        self,
        extracted_data: Dict[str, Any],
        prefix: str = "sih_export",
        payload_bitstring: str = "",
    ) -> Dict[str, str]:
        """Utility wrapper to generate JSON, CSV, and Payload exports in one call."""
        json_file = self.export_json(extracted_data, f"{prefix}_summary.json")
        csv_file = self.export_csv(extracted_data, f"{prefix}_metrics.csv")

        results = {"json_path": json_file, "csv_path": csv_file}

        if payload_bitstring:
            bin_file = self.export_payload_binary(
                payload_bitstring, f"{prefix}_payload.bin"
            )
            results["payload_path"] = bin_file

        return results


if __name__ == "__main__":
    # Sample Test Run
    sample_metrics = {
        "sample_rate": 2000000,
        "center_frequency": 433920000,
        "snr_db": 18.25,
        "bandwidth_hz": 250000,
        "modulation": "QPSK",
        "sync_word_found": True,
        "sync_index": 16,
    }

    sample_payload = "0100000101000100010010010101010001011001"  # "ADITY"

    exporter = SignalExporter(output_dir="sample_exports")
    generated_files = exporter.export_all(
        sample_metrics, prefix="test_run", payload_bitstring=sample_payload
    )

    print("Exports generated successfully:")
    for key, path in generated_files.items():
        print(f" - {key}: {path}")