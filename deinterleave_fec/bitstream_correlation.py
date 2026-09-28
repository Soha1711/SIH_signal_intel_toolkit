"""
SIH 2026 - Signal Intelligence Toolkit
Module 8: Bitstream Correlation & Frame Analysis
Owner: Tirth

Replaces hardcoded string verification with real preamble / sync-word cross-correlation.
Autonomously detects message boundaries, handles phase inversion (180 deg BPSK ambiguity),
tolerates channel noise/bit errors, and cleanly splits headers from payloads.

Conforms to docs/interface_contract.md (Step 6: Bitstream Correlation -> Output).
"""

from typing import Union, Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


# ============================================================
# STANDARD SYNC WORDS / PREAMBLES
# ============================================================

KNOWN_SYNC_WORDS = {
    "SIH_DEFAULT": "10101011",                       # 8-bit (0xAB) used in SIH 2026
    "BARKER_7": "1110010",                            # 7-bit Barker code
    "BARKER_11": "11100010010",                       # 11-bit Barker code
    "BARKER_13": "1111100110101",                     # 13-bit Barker code
    "CCSDS_32": "00011010110011111111110000011101",   # 32-bit ASM (0x1ACFFC1D)
    "AX25_FLAG": "01111110",                          # 8-bit HDLC / AX.25 flag (0x7E)
}


@dataclass
class CorrelationResult:
    """Detailed result of bitstream cross-correlation."""
    found: bool
    sync_index: int                    # Start bit index of sync word in stream
    frame_start: int                   # Bit index where header/payload begins
    sync_word: str                     # Pattern matched
    correlation_score: float           # Normalized score [-1.0, 1.0] (1.0 = exact match)
    bit_errors: int                    # Number of bit errors inside the sync word
    is_inverted: bool                  # True if phase inversion detected (negative correlation peak)
    peak_scores: np.ndarray            # Sliding correlation profile across bitstream
    matched_bits: str = ""             # Actual bits detected in the stream at sync_index

    @property
    def sync_found(self) -> bool:
        return self.found

    def __getitem__(self, item: str):
        if item == "sync_found":
            return self.found
        return getattr(self, item)

    def get(self, item: str, default=None):
        try:
            return self[item]
        except (AttributeError, KeyError):
            return default


# ============================================================
# 1. BITSTREAM CORRELATOR (Autonomous Boundary Detection)
# ============================================================

class BitstreamCorrelator:
    """
    Cross-Correlation Engine for Bitstreams.
    
    Instead of hardcoding message offsets or using brittle exact string searches
    (.find()), this engine computes sliding-window bipolar cross-correlation:
        b_bipolar in {-1, +1},  sync_bipolar in {-1, +1}
        R[i] = sum(b_bipolar[i : i + L] * sync_bipolar) / L
        
    Key Capabilities:
    1. Autonomous Boundary Detection: scans the entire bitstream with leading/trailing noise.
    2. Noise Tolerance: accepts sync words with <= max_bit_errors (or score >= threshold).
    3. Phase Ambiguity Resolution: detects negative peaks (180 deg BPSK phase lock),
       and automatically inverts the bitstream so downstream FEC/decoders receive correct data.
    4. Multi-Pattern Search: checks against candidate sync words (Barker, CCSDS, etc.).
    """

    def __init__(
        self,
        default_sync_pattern: str = "10101011",
        confidence_threshold: float = 0.75,
        max_bit_errors: Optional[int] = None,
        auto_invert: bool = True
    ):
        self.default_sync = default_sync_pattern
        self.confidence_threshold = confidence_threshold
        self.max_bit_errors = max_bit_errors
        self.auto_invert = auto_invert

    @staticmethod
    def to_bipolar(bits: Union[str, np.ndarray, List[int]]) -> np.ndarray:
        """Converts binary 0/1 bits to bipolar -1.0 / +1.0 values."""
        if isinstance(bits, str):
            arr = np.array([1.0 if b == "1" else -1.0 for b in bits], dtype=np.float32)
        else:
            arr = np.where(np.asarray(bits) > 0, 1.0, -1.0).astype(np.float32)
        return arr

    def correlate(
        self,
        bitstream: Union[str, np.ndarray, List[int]],
        sync_pattern: Optional[str] = None
    ) -> CorrelationResult:
        """
        Perform sliding-window cross-correlation against a sync pattern.
        
        Parameters
        ----------
        bitstream : str or np.ndarray
            Incoming demodulated bitstream (may contain leading noise, clock drift, etc.)
        sync_pattern : str, optional
            Sync word to search for. If None, uses self.default_sync.
            
        Returns
        -------
        CorrelationResult
        """
        pattern = sync_pattern or self.default_sync
        
        # Ensure string representation for pattern
        if not isinstance(pattern, str):
            pattern = "".join(map(str, pattern))
            
        L = len(pattern)
        if L == 0:
            raise ValueError("Sync pattern cannot be empty")

        # Convert bitstream to string and bipolar array
        if isinstance(bitstream, str):
            bit_str = bitstream
        else:
            bit_str = "".join(map(str, np.asarray(bitstream, dtype=int)))

        N = len(bit_str)
        if N < L:
            return CorrelationResult(
                found=False,
                sync_index=-1,
                frame_start=-1,
                sync_word=pattern,
                correlation_score=0.0,
                bit_errors=L,
                is_inverted=False,
                peak_scores=np.array([], dtype=np.float32)
            )

        b_bipolar = self.to_bipolar(bit_str)
        s_bipolar = self.to_bipolar(pattern)

        # Sliding dot product across all valid offsets
        num_offsets = N - L + 1
        scores = np.empty(num_offsets, dtype=np.float32)

        for i in range(num_offsets):
            scores[i] = np.dot(b_bipolar[i : i + L], s_bipolar) / L

        # Find maximum positive peak and minimum negative peak
        max_pos_idx = int(np.argmax(scores))
        max_pos_score = float(scores[max_pos_idx])

        min_neg_idx = int(np.argmin(scores))
        min_neg_score = float(scores[min_neg_idx])

        # Prioritize positive direct correlation on tie
        if abs(min_neg_score) > max_pos_score:
            best_idx = min_neg_idx
            best_score = min_neg_score
            abs_best_score = abs(min_neg_score)
            is_inverted_candidate = True
        else:
            best_idx = max_pos_idx
            best_score = max_pos_score
            abs_best_score = max_pos_score
            is_inverted_candidate = False

        # Bit errors corresponding to this score:
        # score = (L - 2*errors) / L  ==>  errors = round(L * (1 - abs_score) / 2)
        detected_errors = int(round(L * (1.0 - abs_best_score) / 2.0))

        # Check detection criteria
        passed_threshold = abs_best_score >= self.confidence_threshold
        if self.max_bit_errors is not None:
            passed_threshold = passed_threshold and (detected_errors <= self.max_bit_errors)

        is_inverted = is_inverted_candidate and passed_threshold
        matched_slice = bit_str[best_idx : best_idx + L] if passed_threshold else ""

        return CorrelationResult(
            found=passed_threshold,
            sync_index=best_idx if passed_threshold else -1,
            frame_start=(best_idx + L) if passed_threshold else -1,
            sync_word=pattern,
            correlation_score=best_score if passed_threshold else 0.0,
            bit_errors=detected_errors if passed_threshold else L,
            is_inverted=is_inverted,
            peak_scores=scores,
            matched_bits=matched_slice
        )

    def correlate_multi_candidate(
        self,
        bitstream: Union[str, np.ndarray, List[int]],
        candidate_patterns: Optional[Dict[str, str]] = None
    ) -> CorrelationResult:
        """
        Scans bitstream against multiple standard sync patterns and returns the best match.
        """
        candidates = candidate_patterns or KNOWN_SYNC_WORDS
        best_res = None
        best_abs_score = -1.0

        for name, pat in candidates.items():
            res = self.correlate(bitstream, sync_pattern=pat)
            if res.found and abs(res.correlation_score) > best_abs_score:
                best_abs_score = abs(res.correlation_score)
                best_res = res

        if best_res is not None and best_res.found:
            return best_res

        # Fallback to default
        return self.correlate(bitstream, sync_pattern=self.default_sync)


# ============================================================
# 2. FRAME PARSER (Header vs. Payload Splitting)
# ============================================================

class FrameParser:
    """
    Splits demodulated bitstream into Header and Payload once the sync word is found.
    
    Supports:
    - Raw Payload Mode (header_length = 0): Entire post-sync stream is payload.
    - Fixed Header Mode (header_length > 0): Specific number of bits allocated to header.
    - Structured Header Mode: Automatically parses standard telemetry fields:
        [Packet Type: 4b | Flags: 4b | Payload Length: 8b | Seq Num: 8b | CRC8: 8b]
    """

    def __init__(
        self,
        header_length_bits: int = 0,
        structured_header: bool = False
    ):
        self.header_length_bits = header_length_bits
        self.structured_header = structured_header

    @staticmethod
    def parse_standard_header(header_bits: str) -> Dict[str, Any]:
        """
        Parses a 32-bit standard RF framing header:
        - Bits 0-3  : Packet Type (int, 0-15)
        - Bits 4-7  : Flags (bit 0: has_fec, bit 1: has_interleaving, bit 2-3: reserved)
        - Bits 8-15 : Payload Length in bytes
        - Bits 16-23: Packet Sequence Number (0-255)
        - Bits 24-31: Header Checksum (CRC-8 or XOR parity)
        """
        if len(header_bits) < 32:
            return {
                "raw_header": header_bits,
                "parsed": False,
                "error": f"Header too short: expected 32 bits, got {len(header_bits)}"
            }

        pkt_type = int(header_bits[0:4], 2)
        flags_val = int(header_bits[4:8], 2)
        payload_byte_len = int(header_bits[8:16], 2)
        seq_num = int(header_bits[16:24], 2)
        checksum = int(header_bits[24:32], 2)

        return {
            "parsed": True,
            "packet_type": pkt_type,
            "flags": {
                "has_fec": bool(flags_val & 0x1),
                "has_interleaving": bool(flags_val & 0x2),
                "raw_flags": flags_val
            },
            "payload_byte_length": payload_byte_len,
            "payload_bit_length": payload_byte_len * 8,
            "sequence_number": seq_num,
            "header_checksum": checksum,
            "raw_header": header_bits
        }

    def split_frame(
        self,
        bitstream: Union[str, np.ndarray],
        correlation_result: CorrelationResult,
        auto_invert: bool = True
    ) -> Dict[str, Any]:
        """
        Splits frame into Header and Payload based on correlation result.
        
        Parameters
        ----------
        bitstream : str or np.ndarray
            Incoming bitstream.
        correlation_result : CorrelationResult
            Result of sync-word correlation.
        auto_invert : bool
            If True and phase inversion was detected, inverts bits automatically.
            
        Returns
        -------
        dict
            Complies with docs/interface_contract.md Step 6.
        """
        if isinstance(bitstream, str):
            bit_str = bitstream
        else:
            bit_str = "".join(map(str, np.asarray(bitstream, dtype=int)))

        if not correlation_result.found:
            return {
                "sync_found": False,
                "header_start": -1,
                "header_end": -1,
                "payload_start": -1,
                "payload_end": -1,
                "sync_word_matched": "",
                "sync_index": -1,
                "correlation_score": 0.0,
                "matched_bits": "",
                "is_inverted": False,
                "header_bits": "",
                "payload_bits": "",
                "header_fields": {},
                "frame_length": 0,
                "payload_length": 0,
                "total_frame_bits": 0,
                "status": "no_sync_word_found"
            }

        # Resolve phase inversion if needed
        if correlation_result.is_inverted and auto_invert:
            # Invert all bits: 0->1, 1->0
            inverted_chars = ["1" if b == "0" else "0" for b in bit_str]
            bit_str = "".join(inverted_chars)

        sync_len = len(correlation_result.sync_word)
        frame_start = correlation_result.sync_index + sync_len

        header_len = 32 if self.structured_header else self.header_length_bits
        header_start = frame_start
        header_end = frame_start + header_len
        payload_start = header_end

        # Slice header
        header_bits = bit_str[header_start:header_end] if header_len > 0 else ""

        # Slice payload
        header_fields = {}
        if self.structured_header and len(header_bits) >= 32:
            header_fields = self.parse_standard_header(header_bits)
            # If payload length is specified in header, truncate payload to exact size
            if header_fields.get("parsed", False):
                bit_len = header_fields["payload_bit_length"]
                payload_end = payload_start + bit_len
                payload_bits = bit_str[payload_start:payload_end]
            else:
                payload_end = len(bit_str)
                payload_bits = bit_str[payload_start:]
        else:
            payload_end = len(bit_str)
            payload_bits = bit_str[payload_start:]

        frame_len = len(header_bits) + len(payload_bits)

        return {
            "sync_found": True,
            "header_start": header_start,
            "header_end": header_end,
            "payload_start": payload_start,
            "payload_end": payload_end,
            "sync_word_matched": correlation_result.sync_word,
            "sync_index": correlation_result.sync_index,
            "correlation_score": correlation_result.correlation_score,
            "matched_bits": correlation_result.matched_bits,
            "is_inverted": correlation_result.is_inverted,
            "header_bits": header_bits,
            "payload_bits": payload_bits,
            "header_fields": header_fields,
            "frame_length": frame_len,
            "payload_length": len(payload_bits),
            "total_frame_bits": frame_len,
            "status": "sync_found"
        }


# ============================================================
# 3. PIPELINE INTEGRATION FUNCTION (Step 6)
# ============================================================

def correlate_and_frame(
    stage_data: Union[dict, str, np.ndarray],
    sync_pattern: str = "10101011",
    header_length_bits: int = 0,
    confidence_threshold: float = 0.75,
    structured_header: bool = False,
    auto_invert: bool = True
) -> dict:
    """
    Main Bitstream Correlation & Framing entry point.
    
    Conforms to docs/interface_contract.md (Step 6).
    Accepts upstream stage dictionary or raw bitstream, dynamically locates the
    sync word boundary, resolves phase ambiguity, and splits header vs payload.
    """
    if isinstance(stage_data, dict):
        base_dict = dict(stage_data)
        bitstream = base_dict.get("decoded_bits", base_dict.get("deinterleaved_bits", base_dict.get("bitstream", "")))
    else:
        base_dict = {}
        bitstream = stage_data

    correlator = BitstreamCorrelator(
        default_sync_pattern=sync_pattern,
        confidence_threshold=confidence_threshold,
        auto_invert=auto_invert
    )
    corr_res = correlator.correlate(bitstream, sync_pattern=sync_pattern)

    parser = FrameParser(
        header_length_bits=header_length_bits,
        structured_header=structured_header
    )
    frame_report = parser.split_frame(bitstream, corr_res, auto_invert=auto_invert)

    # Merge into output dictionary
    output_dict = {**base_dict, **frame_report}
    return output_dict
