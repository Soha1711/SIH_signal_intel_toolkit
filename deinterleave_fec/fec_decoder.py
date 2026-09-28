"""
SIH 2026 - FEC Decoder & De-interleaver Module
Owner: Tirth

Supports:
1. Viterbi hard-decision convolutional decoder (K=3, rate 1/2, polynomials 7 and 5 octal)
2. Hamming(7,4) single-error correcting syndrome decoder (matching dataset_generator.py)
3. De-interleaving integration via BlockInterleaver
4. Interface contract Step 5 (FEC Decoding -> Bitstream Correlation)
"""

from typing import Iterable, Tuple, Union, Optional, List, Dict, Any
import numpy as np
from deinterleave_fec.interleaver import deinterleave_bits, BlockInterleaver


# ============================================================
# CONVOLUTIONAL CODE CONFIGURATION
# ============================================================

CONSTRAINT_LENGTH = 3

# 7 octal = 111 binary, 5 octal = 101 binary
GENERATORS = (0b111, 0b101)

# Number of memory bits
MEMORY = CONSTRAINT_LENGTH - 1

# Number of states in the trellis
NUM_STATES = 2 ** MEMORY


# ============================================================
# HAMMING (7,4) MATRICES & SYNDROME DECODER
# ============================================================

# Generator matrix matching dataset_generator.py (systematic: [I_4 | P])
HAMMING_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
], dtype=int)

# Parity check matrix H = [P^T | I_3]
HAMMING_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1]
], dtype=int)

# Syndrome-to-error-bit lookup map
HAMMING_SYNDROME_MAP = {
    (1, 1, 0): 0,
    (1, 0, 1): 1,
    (0, 1, 1): 2,
    (1, 1, 1): 3,
    (1, 0, 0): 4,
    (0, 1, 0): 5,
    (0, 0, 1): 6,
}


def hamming_7_4_encode_bits(bit_str: str) -> str:
    """Encodes 4-bit data chunks into 7-bit Hamming(7,4) codewords."""
    pad_len = (4 - len(bit_str) % 4) % 4
    bit_str_padded = bit_str + "0" * pad_len
    
    encoded_bits = []
    for i in range(0, len(bit_str_padded), 4):
        data = np.array([int(b) for b in bit_str_padded[i:i+4]], dtype=int)
        codeword = np.dot(data, HAMMING_G) % 2
        encoded_bits.extend(codeword.tolist())
        
    return "".join(map(str, encoded_bits))


def hamming_7_4_decode_bits(
    bit_str: str,
    original_length: Optional[int] = None
) -> Tuple[str, int]:
    """
    Decodes a stream of 7-bit Hamming(7,4) codewords into 4-bit data chunks.
    Performs syndrome decoding to detect and correct any single-bit error per codeword.
    
    Returns
    -------
    decoded_bits : str
        Decoded data bitstream.
    corrected_errors : int
        Total number of bit errors detected and corrected.
    """
    if not bit_str:
        return "", 0

    usable_len = (len(bit_str) // 7) * 7
    corrected_errors = 0
    decoded_chunks = []

    for i in range(0, usable_len, 7):
        chunk = np.array([int(b) for b in bit_str[i:i+7]], dtype=int)
        syndrome = tuple(int(s) for s in (np.dot(HAMMING_H, chunk) % 2))
        
        # Check for error
        if syndrome != (0, 0, 0):
            if syndrome in HAMMING_SYNDROME_MAP:
                error_pos = HAMMING_SYNDROME_MAP[syndrome]
                chunk[error_pos] ^= 1
                corrected_errors += 1
            else:
                # Uncorrectable multi-bit error
                pass

        # First 4 bits are the data bits in systematic form
        decoded_chunks.extend(chunk[:4].tolist())

    decoded_bits = "".join(map(str, decoded_chunks))
    if original_length is not None and original_length < len(decoded_bits):
        decoded_bits = decoded_bits[:original_length]

    return decoded_bits, corrected_errors


# ============================================================
# CONVOLUTIONAL / VITERBI HELPER FUNCTIONS
# ============================================================

def parity(value: int) -> int:
    """Returns the parity of the set bits."""
    return value.bit_count() % 2


def encoder_output(state: int, input_bit: int) -> Tuple[int, int]:
    """Calculate the two coded output bits for a given encoder state and input bit."""
    shift_register = (state << 1) | input_bit
    output_1 = parity(shift_register & GENERATORS[0])
    output_2 = parity(shift_register & GENERATORS[1])
    return output_1, output_2


def next_state(state: int, input_bit: int) -> int:
    """Calculate the next convolutional encoder state."""
    return ((state << 1) | input_bit) & ((1 << MEMORY) - 1)


def hamming_distance(received: Tuple[int, int], expected: Tuple[int, int]) -> int:
    """Calculate the Hamming distance between two pairs of coded bits."""
    return (received[0] != expected[0]) + (received[1] != expected[1])


# ============================================================
# VITERBI DECODER
# ============================================================

def viterbi_decode(
    received_bits: Union[Iterable[int], str],
    terminate: bool = True
) -> str:
    """
    Decode a rate-1/2 convolutional bitstream using hard-decision Viterbi decoding.
    
    Parameters
    ----------
    received_bits:
        Iterable or string containing 0/1 coded bits.
    terminate:
        If True, assume encoder was terminated with two zero tail bits and force final state 0.
        
    Returns
    -------
    str:
        Decoded information bits.
    """
    bits = [int(bit) for bit in received_bits]

    if len(bits) % 2 != 0:
        raise ValueError("Viterbi decoder requires an even number of coded bits.")

    if not bits:
        return ""

    num_steps = len(bits) // 2
    INF = 10**9

    path_metric = [INF for _ in range(NUM_STATES)]
    path_metric[0] = 0
    survivors = []

    for step in range(num_steps):
        received_pair = (bits[2 * step], bits[2 * step + 1])
        new_metric = [INF for _ in range(NUM_STATES)]
        new_survivor = [None for _ in range(NUM_STATES)]

        for state in range(NUM_STATES):
            current_metric = path_metric[state]
            if current_metric >= INF:
                continue

            for input_bit in (0, 1):
                output = encoder_output(state, input_bit)
                destination = next_state(state, input_bit)
                branch_metric = hamming_distance(received_pair, output)
                total_metric = current_metric + branch_metric

                if total_metric < new_metric[destination]:
                    new_metric[destination] = total_metric
                    new_survivor[destination] = (state, input_bit)

        path_metric = new_metric
        survivors.append(new_survivor)

    if terminate:
        final_state = 0
    else:
        final_state = min(range(NUM_STATES), key=lambda state: path_metric[state])

    decoded_reverse = []
    state = final_state

    for step in range(num_steps - 1, -1, -1):
        survivor = survivors[step][state]
        if survivor is None:
            raise RuntimeError("Viterbi traceback failed.")
        previous_state, input_bit = survivor
        decoded_reverse.append(input_bit)
        state = previous_state

    decoded_bits = list(reversed(decoded_reverse))

    if terminate and len(decoded_bits) >= MEMORY:
        decoded_bits = decoded_bits[:-MEMORY]

    return "".join(str(bit) for bit in decoded_bits)


# ============================================================
# ASCII CONVERSION HELPER
# ============================================================

def bits_to_ascii(bit_string: str, allow_truncated: bool = True) -> str:
    """
    Convert a binary string into ASCII text.
    If allow_truncated is True, decodes complete 8-bit bytes and ignores remaining bits.
    """
    usable_len = (len(bit_string) // 8) * 8
    characters = []

    for index in range(0, usable_len, 8):
        byte = bit_string[index : index + 8]
        characters.append(chr(int(byte, 2)))

    return "".join(characters)


# ============================================================
# UNIFIED FEC DECODER CLASS
# ============================================================

class FECDecoder:
    """
    Unified FEC Decoder supporting:
    - "hamming": Hamming (7,4) code
    - "viterbi": Convolutional rate-1/2 code
    - "none": Pass-through uncoded
    """

    def __init__(self, scheme: str = "hamming"):
        self.scheme = scheme.lower().strip()

    def decode(
        self,
        bits: Union[str, np.ndarray],
        scheme: Optional[str] = None,
        original_length: Optional[int] = None
    ) -> Tuple[str, int]:
        selected_scheme = (scheme or self.scheme).lower().strip()
        bit_str = bits if isinstance(bits, str) else "".join(map(str, np.asarray(bits, dtype=int)))

        if selected_scheme in ("hamming", "hamming_7_4"):
            return hamming_7_4_decode_bits(bit_str, original_length=original_length)
        elif selected_scheme in ("viterbi", "convolutional"):
            decoded = viterbi_decode(bit_str, terminate=True)
            return decoded, 0
        elif selected_scheme in ("none", "raw", "uncoded"):
            return bit_str, 0
        else:
            raise ValueError(f"Unsupported FEC scheme: {selected_scheme}")


# ============================================================
# PIPELINE INTERFACE CONTRACT (Step 5)
# ============================================================

def process_fec_decoding(
    stage_data: dict,
    fec_scheme: str = "hamming",
    original_length: Optional[int] = None
) -> dict:
    """
    Interface function fulfilling docs/interface_contract.md Step 5:
    De-interleaving -> FEC Decoding -> Bitstream Correlation.
    
    Parameters
    ----------
    stage_data : dict
        Dict from de-interleaver containing 'deinterleaved_bits'.
    fec_scheme : str
        "hamming" | "viterbi" | "none"
        
    Returns
    -------
    dict
        Extends stage_data with:
        'decoded_bits': np.ndarray (uint8)
        'fec_scheme': str
        'error_count': int
        'decoded_bitstring': str
        'ascii_message': str
    """
    result = dict(stage_data)
    deint = stage_data.get("deinterleaved_bits", stage_data.get("bitstream", ""))

    if isinstance(deint, np.ndarray):
        bit_str = "".join(map(str, deint.astype(int)))
    else:
        bit_str = str(deint)

    decoder = FECDecoder(scheme=fec_scheme)
    decoded_str, err_count = decoder.decode(bit_str, original_length=original_length)

    decoded_arr = np.array([int(b) for b in decoded_str], dtype=np.uint8) if decoded_str else np.array([], dtype=np.uint8)

    result["decoded_bits"] = decoded_arr
    result["decoded_bitstring"] = decoded_str
    result["fec_scheme"] = fec_scheme
    result["error_count"] = err_count
    result["ascii_message"] = bits_to_ascii(decoded_str)

    return result


# ============================================================
# STANDALONE VERIFICATION TESTS
# ============================================================

def test_deinterleaver():
    """Verify block interleaver / de-interleaver round-trip."""
    original_bits = (
        "1011001010101011"
        "1110001110101010"
        "0101010111110000"
    )
    interleaver = BlockInterleaver(cols=8)
    interleaved, _ = interleaver.interleave(original_bits)
    recovered = interleaver.deinterleave(interleaved, original_length=len(original_bits))
    assert recovered == original_bits, "De-interleaver failed to recover original bits."
    print("[PASS] Block Interleaver & De-interleaver verified.")


def test_hamming_fec():
    """Verify Hamming(7,4) encoder and decoder with injected errors."""
    data = "0101001101001001"  # 16 bits
    encoded = hamming_7_4_encode_bits(data)
    
    # Inject errors in first and second codeword
    noisy = list(encoded)
    noisy[1] = "1" if noisy[1] == "0" else "0"
    noisy[10] = "1" if noisy[10] == "0" else "0"
    noisy_str = "".join(noisy)

    decoded, errors = hamming_7_4_decode_bits(noisy_str, original_length=len(data))
    assert decoded == data, f"Hamming decode mismatch: expected {data}, got {decoded}"
    assert errors == 2, f"Expected 2 corrected errors, got {errors}"
    print("[PASS] Hamming(7,4) FEC error correction verified.")


def test_viterbi_fec():
    """Verify Viterbi decoder with injected errors."""
    data = "01010011"  # 'S'
    state = 0
    encoded = []
    input_bits = data + "0" * MEMORY

    for bit_char in input_bits:
        bit = int(bit_char)
        shift_register = (state << 1) | bit
        out_1 = parity(shift_register & GENERATORS[0])
        out_2 = parity(shift_register & GENERATORS[1])
        encoded.extend([out_1, out_2])
        state = ((state << 1) | bit) & ((1 << MEMORY) - 1)

    # Flip 1 bit
    received = encoded.copy()
    received[3] ^= 1

    decoded = viterbi_decode(received, terminate=True)
    assert decoded == data, f"Viterbi decode mismatch: expected {data}, got {decoded}"
    print("[PASS] Viterbi convolutional decoder verified.")


if __name__ == "__main__":
    test_deinterleaver()
    test_hamming_fec()
    test_viterbi_fec()
    print("All FEC and de-interleaver tests passed successfully.")
