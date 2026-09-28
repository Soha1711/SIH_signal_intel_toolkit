"""
SIH 2026 - Unit and Integration Tests for De-interleaving and Bitstream Correlation
Owner: Tirth (Module 6: De-interleaving, Module 8: Bitstream Correlation & Framing)

Covers the 7 required test cases:
1. Interleave/de-interleave round trip (with padding handling)
2. Sync at beginning (index 0)
3. Sync in middle
4. Noise before sync (autonomous boundary detection)
5. Sync with 1–2 bit errors (noise tolerance)
6. Sync not found
7. Header/payload extraction (SYNC -> HEADER -> PAYLOAD)
+ Independent test against Person A's synthetic interleaved files (.iq)
"""

import json
from pathlib import Path
import numpy as np
import pytest

from deinterleave_fec.interleaver import (
    BlockInterleaver,
    interleave,
    deinterleave,
    interleave_bits,
    deinterleave_bits,
)
from deinterleave_fec.bitstream_correlation import (
    BitstreamCorrelator,
    FrameParser,
    correlate_and_frame,
)
from deinterleave_fec.fec_decoder import (
    hamming_7_4_decode_bits,
    bits_to_ascii,
)
from demodulation.robust_demodulator import demodulate_symbols


# ============================================================
# 1. TEST: Interleave / De-interleave Round Trip
# ============================================================

def test_interleave_deinterleave_round_trip():
    """
    Test 1: original -> interleave -> deinterleave -> original.
    Must correctly reverse block interleaving and handle padding cleanly.
    """
    # Test case A: length perfectly divisible by 8 (16 bits)
    orig_16 = "1100101001110010"
    interleaved_16 = interleave(orig_16, cols=8)
    recovered_16 = deinterleave(interleaved_16, cols=8, original_length=len(orig_16))
    assert recovered_16 == orig_16, f"Mismatch on 16 bits: {recovered_16} != {orig_16}"

    # Test case B: length not divisible by 8 (13 bits, 3 padding bits added)
    orig_13 = "1010101110001"
    interleaved_13 = interleave(orig_13, cols=8)
    assert len(interleaved_13) == 16  # padded to multiple of 8
    recovered_13 = deinterleave(interleaved_13, cols=8, original_length=len(orig_13))
    assert recovered_13 == orig_13, f"Mismatch on padded 13 bits: {recovered_13} != {orig_13}"

    # Test case C: numpy array input
    orig_arr = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1], dtype=np.uint8)
    interleaved_arr = interleave(orig_arr, cols=8)
    recovered_arr = deinterleave(interleaved_arr, cols=8, original_length=len(orig_arr))
    assert np.array_equal(recovered_arr, orig_arr), "Numpy round-trip failed"


# ============================================================
# 2. TEST: Sync at Beginning
# ============================================================

def test_sync_at_beginning():
    """
    Test 2: Sync word at beginning of stream (index 0).
    """
    sync_word = "10101011"
    payload = "0100100001101001"  # 'Hi'
    stream = sync_word + payload

    correlator = BitstreamCorrelator(default_sync_pattern=sync_word)
    res = correlator.correlate(stream)

    assert res.sync_found is True
    assert res.sync_index == 0
    assert res.correlation_score == 1.0
    assert res.matched_bits == sync_word
    assert res.bit_errors == 0


# ============================================================
# 3. TEST: Sync in Middle
# ============================================================

def test_sync_in_middle():
    """
    Test 3: Sync word located in the middle of a bitstream.
    Ensures correlator does NOT assume sync starts at index 0.
    """
    sync_word = "10101011"
    prefix = "00011010"     # 8 bits prefix
    suffix = "11100101001"  # 11 bits suffix
    stream = prefix + sync_word + suffix

    correlator = BitstreamCorrelator(default_sync_pattern=sync_word)
    res = correlator.correlate(stream)

    assert res.sync_found is True
    assert res.sync_index == len(prefix)  # index 8
    assert res.correlation_score == 1.0
    assert res.matched_bits == sync_word


# ============================================================
# 4. TEST: Noise Before Sync (Autonomous Boundary Detection)
# ============================================================

def test_noise_before_sync():
    """
    Test 4: Random noise before sync word.
    The system must find the message boundary itself without being told the index.
    """
    sync_word = "10101011"
    payload = "010100110100100101001000"  # 'SIH'
    
    # 37 random bits of channel noise
    np.random.seed(42)
    noise_prefix = "".join(str(np.random.randint(0, 2)) for _ in range(37))
    noise_suffix = "11001100"
    noisy_stream = noise_prefix + sync_word + payload + noise_suffix

    correlator = BitstreamCorrelator(default_sync_pattern=sync_word, confidence_threshold=0.8)
    res = correlator.correlate(noisy_stream)

    assert res.sync_found is True
    assert res.sync_index == 37
    assert res.correlation_score == 1.0
    assert res.matched_bits == sync_word

    # Extract payload starting after sync word
    extracted_payload = noisy_stream[res.frame_start : res.frame_start + len(payload)]
    assert extracted_payload == payload


# ============================================================
# 5. TEST: Sync with 1–2 Bit Errors (Noise Tolerance)
# ============================================================

def test_sync_with_1_to_2_bit_errors():
    """
    Test 5: Sync word containing 1 or 2 flipped bits due to channel noise.
    Real cross-correlation must tolerate bit errors while exact string search fails.
    """
    sync_word = "10101011"  # 8 bits
    prefix = "00001111"      # 8 bits

    # Case A: 1 bit error ('10101011' -> '10001011', index 2 flipped)
    sync_1err = "10001011"
    stream_1err = prefix + sync_1err + "01010101"
    correlator_1err = BitstreamCorrelator(default_sync_pattern=sync_word, confidence_threshold=0.7)
    res_1err = correlator_1err.correlate(stream_1err)

    assert res_1err.sync_found is True
    assert res_1err.sync_index == 8
    assert res_1err.bit_errors == 1
    assert res_1err.correlation_score == 0.75  # (8 - 2)/8 = 0.75
    assert res_1err.matched_bits == sync_1err

    # Case B: 2 bit errors ('10101011' -> '10000011', indices 2 and 4 flipped)
    sync_2err = "10000011"
    stream_2err = prefix + sync_2err + "00000000"
    correlator_2err = BitstreamCorrelator(
        default_sync_pattern=sync_word,
        confidence_threshold=0.5,
        max_bit_errors=2
    )
    res_2err = correlator_2err.correlate(stream_2err)

    assert res_2err.sync_found is True
    assert res_2err.sync_index == 8
    assert res_2err.bit_errors == 2
    assert res_2err.correlation_score == 0.5   # (8 - 4)/8 = 0.50
    assert res_2err.matched_bits == sync_2err


# ============================================================
# 6. TEST: Sync Not Found
# ============================================================

def test_sync_not_found():
    """
    Test 6: Bitstream does not contain the sync word.
    Must return sync_found=False without throwing an exception or falsely matching.
    """
    sync_word = "10101011"
    # Bitstream with no match to sync_word
    no_sync_stream = "00000000000000001111111100000000"

    correlator = BitstreamCorrelator(default_sync_pattern=sync_word, confidence_threshold=0.8)
    res = correlator.correlate(no_sync_stream)

    assert res.sync_found is False
    assert res.sync_index == -1
    assert res.correlation_score == 0.0

    parser = FrameParser()
    frame = parser.split_frame(no_sync_stream, res)
    assert frame["sync_found"] is False
    assert frame["sync_index"] == -1
    assert frame["payload_bits"] == ""
    assert frame["frame_length"] == 0


# ============================================================
# 7. TEST: Header and Payload Extraction (SYNC -> HEADER -> PAYLOAD)
# ============================================================

def test_header_payload_extraction():
    """
    Test 7: Message framing: SYNC -> HEADER -> PAYLOAD.
    Validates extracting header bits, payload bits, frame length,
    sync position, and correlation score without assuming hardcoded offsets.
    """
    sync_word = "10101011"             # 8 bits
    header_bits = "1111000010100101"     # 16 bits header (e.g. Length + Flags)
    payload_bits = "0100000101000010"    # 16 bits payload ('AB')
    
    # Prepend 12 noise bits
    noise_prefix = "010101100011"
    full_stream = noise_prefix + sync_word + header_bits + payload_bits

    correlator = BitstreamCorrelator(default_sync_pattern=sync_word)
    corr_res = correlator.correlate(full_stream)
    assert corr_res.sync_found is True
    assert corr_res.sync_index == 12

    # FrameParser configured with 16-bit header
    parser = FrameParser(header_length_bits=16)
    frame = parser.split_frame(full_stream, corr_res)

    assert frame["sync_found"] is True
    assert frame["sync_index"] == 12
    assert frame["correlation_score"] == 1.0
    assert frame["header_bits"] == header_bits
    assert frame["payload_bits"] == payload_bits
    assert frame["frame_length"] == len(header_bits) + len(payload_bits)
    assert frame["payload_length"] == len(payload_bits)
    assert frame["header_start"] == 12 + 8
    assert frame["header_end"] == 12 + 8 + 16
    assert frame["payload_start"] == 12 + 8 + 16


# ============================================================
# 8. TEST: Independent Validation Against Person A's Synthetic Files
# ============================================================

def test_person_a_synthetic_files():
    """
    Validation against Person A's actual interleaved synthetic dataset files (.iq).
    Flow:
        Person A's synthetic file (.iq)
                 ↓
          Demodulate IQ
                 ↓
        CORRELATE SYNC WORD (Autonomously finds boundary)
                 ↓
        SPLIT HEADER & PAYLOAD
                 ↓
        DE-INTERLEAVE PAYLOAD (Cols=8)
                 ↓
        FEC DECODE (Hamming 7,4)
                 ↓
        RECOVER PAYLOAD ('SIH2026')
    """
    manifest_path = Path("synthetic_dataset/manifest.json")
    if not manifest_path.exists():
        pytest.skip("synthetic_dataset/manifest.json not present")

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Pick representative interleaved test signals across BPSK & QPSK from Person A
    test_files = [
        "sig_022_BPSK_snr30dB_fec0_itr1.iq",  # BPSK, 30dB, no FEC, interleaved
        "sig_024_BPSK_snr30dB_fec1_itr1.iq",  # BPSK, 30dB, Hamming FEC, interleaved
        "sig_046_QPSK_snr30dB_fec0_itr1.iq",  # QPSK, 30dB, no FEC, interleaved
        "sig_048_QPSK_snr30dB_fec1_itr1.iq",  # QPSK, 30dB, Hamming FEC, interleaved
        "sig_018_BPSK_snr20dB_fec0_itr1.iq",  # BPSK, 20dB, no FEC, interleaved
        "sig_020_BPSK_snr20dB_fec1_itr1.iq",  # BPSK, 20dB, Hamming FEC, interleaved
    ]

    record_map = {r["file_name"]: r for r in records}
    correlator = BitstreamCorrelator(default_sync_pattern="10101011", confidence_threshold=0.75)

    for fname in test_files:
        rec = record_map.get(fname)
        if not rec:
            continue
        iq_file = Path("synthetic_dataset") / fname
        if not iq_file.exists():
            continue

        iq_samples = np.fromfile(iq_file, dtype=np.complex64)
        demod_bits = demodulate_symbols(iq_samples, rec["modulation"], samples_per_symbol=100)

        # 1. Sync Correlator finds boundary
        corr = correlator.correlate(demod_bits)
        assert corr.sync_found is True, f"Failed sync detection on {fname}"

        # 2. Split frame
        frame = FrameParser(header_length_bits=0).split_frame(demod_bits, corr)
        raw_payload = frame["payload_bits"]

        # 3. De-interleave payload
        deint_payload = deinterleave(raw_payload, cols=8)

        # 4. FEC decode if present
        if rec["has_fec"]:
            decoded_bits, _ = hamming_7_4_decode_bits(deint_payload, original_length=56)
        else:
            decoded_bits = deint_payload[:56]

        # 5. Verify payload recovery
        recovered_text = bits_to_ascii(decoded_bits)
        assert recovered_text == rec["raw_payload_text"], (
            f"Recovery failed on {fname}: expected {rec['raw_payload_text']}, got {recovered_text}"
        )
