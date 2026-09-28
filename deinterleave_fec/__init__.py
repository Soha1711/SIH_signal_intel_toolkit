"""
SIH 2026 - De-interleaving, FEC Decoding, and Bitstream Correlation Toolkit
Owner: Tirth
"""

from deinterleave_fec.interleaver import (
    BlockInterleaver,
    ConvolutionalInterleaver,
    DiagonalInterleaver,
    PseudoRandomInterleaver,
    interleave_bits,
    deinterleave_bits,
    process_deinterleaving,
)

from deinterleave_fec.bitstream_correlation import (
    BitstreamCorrelator,
    FrameParser,
    CorrelationResult,
    correlate_and_frame,
    KNOWN_SYNC_WORDS,
)

from deinterleave_fec.fec_decoder import (
    FECDecoder,
    hamming_7_4_encode_bits,
    hamming_7_4_decode_bits,
    viterbi_decode,
    bits_to_ascii,
    process_fec_decoding,
)

__all__ = [
    # Interleaving
    "BlockInterleaver",
    "ConvolutionalInterleaver",
    "DiagonalInterleaver",
    "PseudoRandomInterleaver",
    "interleave_bits",
    "deinterleave_bits",
    "process_deinterleaving",
    # Correlation & Framing
    "BitstreamCorrelator",
    "FrameParser",
    "CorrelationResult",
    "correlate_and_frame",
    "KNOWN_SYNC_WORDS",
    # FEC
    "FECDecoder",
    "hamming_7_4_encode_bits",
    "hamming_7_4_decode_bits",
    "viterbi_decode",
    "bits_to_ascii",
    "process_fec_decoding",
]
