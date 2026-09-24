import numpy as np
from typing import Dict, Any, Tuple

class PayloadProcessor:
    def __init__(self, bitstream: str = ""):
        self.bitstream = bitstream

    def demodulate_bits(self, signal, samples_per_symbol: int = 1) -> str:
        """
        Slice each symbol's phase sign into a bit.

        samples_per_symbol: how many raw I/Q samples make up one
        transmitted symbol. Slicing every raw sample (the default,
        samples_per_symbol=1) only produces a meaningful bitstream if
        the signal truly has one sample per symbol — for any real
        captured signal (oversampled at the ADC/SDR) this must be set
        to sample_rate / symbol_rate, or the resulting "bits" are
        essentially noise and any sync-word match against them is a
        coincidence, not a real detection.
        """
        if signal is None or len(signal) == 0:
            return ""

        signal = np.asarray(signal)
        samples_per_symbol = max(1, int(samples_per_symbol))

        if samples_per_symbol > 1:
            usable_len = (len(signal) // samples_per_symbol) * samples_per_symbol
            if usable_len == 0:
                return ""
            # Average each block of samples_per_symbol raw samples down to
            # one symbol value before slicing, instead of slicing every
            # raw sample individually.
            blocks = signal[:usable_len].reshape(-1, samples_per_symbol)
            signal = blocks.mean(axis=1)

        if np.iscomplexobj(signal):
            angles = np.angle(signal)
            return "".join(["1" if a > 0 else "0" for a in angles])
        else:
            return "".join(["1" if s > 0 else "0" for s in signal])

    def find_sync_word(self, sync_pattern: str = "10101011") -> Tuple[int, str]:
        sync_index = self.bitstream.find(sync_pattern)
        if sync_index != -1:
            return sync_index, self.bitstream[sync_index + len(sync_pattern):]
        return -1, self.bitstream

    def decode_ascii_payload(self, bitstring: str) -> str:
        chars = []
        for i in range(0, len(bitstring) - len(bitstring) % 8, 8):
            byte_chunk = bitstring[i : i + 8]
            chars.append(chr(int(byte_chunk, 2)))
        return "".join(chars)

    def extract_full_frame(self, sync_pattern: str = "10101011") -> Dict[str, Any]:
        sync_idx, raw_payload = self.find_sync_word(sync_pattern)
        sync_found = sync_idx != -1
        text_decoded = ""
        if sync_found and len(raw_payload) >= 8:
            try:
                text_decoded = self.decode_ascii_payload(raw_payload)
            except Exception:
                text_decoded = "Unable to decode ASCII"
        return {
            "sync_found": sync_found,
            "raw_payload_bits": raw_payload,
            "decoded_text": text_decoded,
        }

    def extract_payload(
        self, signal, sync_pattern: str = "10101011", samples_per_symbol: int = 1
    ) -> Tuple[str, str]:
        if isinstance(signal, (list, np.ndarray)):
            self.bitstream = self.demodulate_bits(signal, samples_per_symbol=samples_per_symbol)
        elif isinstance(signal, str):
            self.bitstream = signal
        frame = self.extract_full_frame(sync_pattern=sync_pattern)
        return frame["raw_payload_bits"], frame["decoded_text"]