from typing import Dict, Any, Tuple


class PayloadProcessor:
    """Processes demodulated bitstreams, locates sync words/preambles,

    and extracts structured header and payload data.
    """

    def __init__(self, bitstream: str):
        self.bitstream = bitstream

    def find_sync_word(self, sync_pattern: str = "10101011") -> Tuple[int, str]:
        """Performs sliding window correlation to detect the sync word index."""
        sync_index = self.bitstream.find(sync_pattern)
        if sync_index != -1:
            # Return index and remaining payload bits after sync pattern
            payload_bits = self.bitstream[sync_index + len(sync_pattern) :]
            return sync_index, payload_bits
        return -1, self.bitstream

    def decode_ascii_payload(self, bitstring: str) -> str:
        """Converts valid 8-bit binary strings to readable ASCII text."""
        chars = []
        # Process in chunks of 8 bits
        for i in range(0, len(bitstring) - len(bitstring) % 8, 8):
            byte_chunk = bitstring[i : i + 8]
            chars.append(chr(int(byte_chunk, 2)))
        return "".join(chars)

    def extract_full_frame(
        self, sync_pattern: str = "10101011"
    ) -> Dict[str, Any]:
        """Extracts sync word metrics and returns decoded payload information."""
        sync_idx, raw_payload = self.find_sync_word(sync_pattern)
        sync_found = sync_idx != -1

        text_decoded = ""
        if sync_found and len(raw_payload) >= 8:
            try:
                text_decoded = self.decode_ascii_payload(raw_payload)
            except Exception:
                text_decoded = "Unable to decode ASCII"

        return {
            "sync_word_used": sync_pattern,
            "sync_found": sync_found,
            "sync_index": sync_idx,
            "payload_length_bits": len(raw_payload),
            "raw_payload_bits": raw_payload,
            "decoded_text": text_decoded,
        }


if __name__ == "__main__":
    # Test sample with sync pattern '10101011' + ASCII 'SIH2026'
    # ASCII 'S' = 01010011, 'I' = 01001001, 'H' = 01001000
    preamble = "00000000"  # Noise/Preamble bits
    sync = "10101011"  # Sync Word
    payload = "010100110100100101001000"  # ASCII Payload ('SIH')

    test_stream = preamble + sync + payload

    processor = PayloadProcessor(test_stream)
    results = processor.extract_full_frame(sync_pattern="10101011")

    print("Bitstream Processing Results:")
    for key, value in results.items():
        print(f" - {key}: {value}")