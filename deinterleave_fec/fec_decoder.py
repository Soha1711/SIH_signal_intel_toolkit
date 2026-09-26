"""
SIH 2026 - Viterbi FEC Decoder

This module decodes the rate-1/2 convolutional code
used by dataset_generator.py.

Encoder configuration:
    Constraint length = 3
    Generator 1       = 7 octal = 111 binary
    Generator 2       = 5 octal = 101 binary
    Code rate         = 1/2
    Tail bits         = 2

Receiver flow:

    Demodulated coded bits
            ↓
       De-interleaving
            ↓
       Viterbi decoder
            ↓
       Original data bits
            ↓
           ASCII
"""

from typing import Iterable, Tuple


# ============================================================
# CONVOLUTIONAL CODE CONFIGURATION
# ============================================================

CONSTRAINT_LENGTH = 3

# 7 octal = 111 binary
# 5 octal = 101 binary
GENERATORS = (0b111, 0b101)

# Number of memory bits.
MEMORY = CONSTRAINT_LENGTH - 1

# Number of states in the trellis.
NUM_STATES = 2 ** MEMORY


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parity(value: int) -> int:
    """
    Returns the parity of the set bits.

    Example:

        0b101 -> two 1s -> parity 0
        0b111 -> three 1s -> parity 1
    """

    return value.bit_count() % 2


def encoder_output(
    state: int,
    input_bit: int
) -> Tuple[int, int]:
    """
    Calculate the two coded output bits for a given
    encoder state and input bit.

    This must exactly match the encoder used by
    dataset_generator.py.
    """

    # Same register construction as the encoder.
    shift_register = (state << 1) | input_bit

    output_1 = parity(
        shift_register & GENERATORS[0]
    )

    output_2 = parity(
        shift_register & GENERATORS[1]
    )

    return output_1, output_2


def next_state(
    state: int,
    input_bit: int
) -> int:
    """
    Calculate the next convolutional encoder state.
    """

    return (
        ((state << 1) | input_bit)
        & ((1 << MEMORY) - 1)
    )


def hamming_distance(
    received: Tuple[int, int],
    expected: Tuple[int, int]
) -> int:
    """
    Calculate the Hamming distance between
    two pairs of coded bits.
    """

    return (
        (received[0] != expected[0])
        + (received[1] != expected[1])
    )

# ============================================================
# DE-INTERLEAVING
# ============================================================

def deinterleave_bits(
    bit_str: str,
    cols: int = 8,
    original_length: int | None = None
) -> str:
    """
    Reverse the block interleaver used by
    dataset_generator.py.

    The transmitter performs:

        bits
          ↓
        rows × cols matrix
          ↓
        transpose
          ↓
        read column-wise

    This function performs the reverse operation.

    Parameters
    ----------
    bit_str:
        Interleaved bit string.

    cols:
        Number of interleaver columns.
        The generator uses 8.

    original_length:
        Original number of bits before interleaver
        padding was added.

        If supplied, padding bits are removed.

    Returns
    -------
    str:
        De-interleaved bit string.
    """

    if not bit_str:
        return ""

    if len(bit_str) % cols != 0:
        raise ValueError(
            "Interleaved bit length must be divisible "
            "by the number of columns."
        )

    rows = len(bit_str) // cols

    # The transmitter created:
    #
    #     rows × cols
    #
    # and then transposed it.
    #
    # Therefore the received interleaved stream is
    # reconstructed as:
    #
    #     cols × rows

    matrix = [
        [
            0 for _ in range(rows)
        ]
        for _ in range(cols)
    ]

    index = 0

    for column in range(cols):

        for row in range(rows):

            matrix[column][row] = int(
                bit_str[index]
            )

            index += 1

    # Transpose back to the original
    # rows × cols arrangement.
    original_matrix = [
        [
            0 for _ in range(cols)
        ]
        for _ in range(rows)
    ]

    for row in range(rows):

        for column in range(cols):

            original_matrix[row][column] = (
                matrix[column][row]
            )

    # Read row-wise to reconstruct the
    # original bit stream.
    bits = []

    for row in range(rows):

        for column in range(cols):

            bits.append(
                str(
                    original_matrix[row][column]
                )
            )

    result = "".join(bits)

    # Remove padding if the original length
    # is known.
    if original_length is not None:

        result = result[:original_length]

    return result

# ============================================================
# VITERBI DECODER
# ============================================================

def viterbi_decode(
    received_bits: Iterable[int],
    terminate: bool = True
) -> str:
    """
    Decode a rate-1/2 convolutional bitstream using
    hard-decision Viterbi decoding.

    Parameters
    ----------
    received_bits:
        Iterable containing 0/1 coded bits.

    terminate:
        If True, assume the encoder was terminated with
        two zero tail bits and force the final state to 0.

    Returns
    -------
    str:
        Decoded information bits.

    Example
    -------
    Original:
        01001001

    Encoded:
        convolutional encoder

    Received:
        encoded bits with possible errors

    Decoded:
        01001001
    """

    # --------------------------------------------------------
    # Convert input to a clean list of integers.
    # --------------------------------------------------------

    bits = [
        int(bit)
        for bit in received_bits
    ]

    # --------------------------------------------------------
    # A rate-1/2 code produces two received bits
    # for every one input bit.
    # --------------------------------------------------------

    if len(bits) % 2 != 0:
        raise ValueError(
            "Viterbi decoder requires an even number "
            "of coded bits."
        )

    if not bits:
        return ""

    # Number of trellis steps.
    num_steps = len(bits) // 2

    # --------------------------------------------------------
    # Trellis storage
    #
    # path_metric[state]
    #     = best metric reaching that state.
    #
    # survivor[state]
    #     = previous state and input bit used
    #       to reach the state.
    # --------------------------------------------------------

    INF = 10**9

    path_metric = [
        INF
        for _ in range(NUM_STATES)
    ]

    # Encoder always starts at state 0.
    path_metric[0] = 0

    survivors = []

    # --------------------------------------------------------
    # Forward pass through the trellis
    # --------------------------------------------------------

    for step in range(num_steps):

        received_pair = (
            bits[2 * step],
            bits[2 * step + 1]
        )

        new_metric = [
            INF
            for _ in range(NUM_STATES)
        ]

        new_survivor = [
            None
            for _ in range(NUM_STATES)
        ]

        # Examine every possible previous state.
        for state in range(NUM_STATES):

            current_metric = path_metric[state]

            if current_metric >= INF:
                continue

            # Each state has two possible transitions:
            #
            # input = 0
            # input = 1
            for input_bit in (0, 1):

                output = encoder_output(
                    state,
                    input_bit
                )

                destination = next_state(
                    state,
                    input_bit
                )

                branch_metric = hamming_distance(
                    received_pair,
                    output
                )

                total_metric = (
                    current_metric
                    + branch_metric
                )

                # Keep the best path.
                if total_metric < new_metric[destination]:

                    new_metric[destination] = (
                        total_metric
                    )

                    new_survivor[destination] = (
                        state,
                        input_bit
                    )

        path_metric = new_metric

        survivors.append(
            new_survivor
        )

    # --------------------------------------------------------
    # Choose final state
    # --------------------------------------------------------

    if terminate:

        # Encoder added zero tail bits.
        # Therefore it should finish in state 0.

        final_state = 0

    else:

        # If no termination is assumed, choose the state
        # having the smallest accumulated metric.
        final_state = min(
            range(NUM_STATES),
            key=lambda state: path_metric[state]
        )

    # --------------------------------------------------------
    # Traceback
    # --------------------------------------------------------

    decoded_reverse = []

    state = final_state

    for step in range(
        num_steps - 1,
        -1,
        -1
    ):

        survivor = survivors[step][state]

        if survivor is None:
            raise RuntimeError(
                "Viterbi traceback failed."
            )

        previous_state, input_bit = survivor

        decoded_reverse.append(
            input_bit
        )

        state = previous_state

    # Reverse because traceback works backwards.
    decoded_bits = list(
        reversed(decoded_reverse)
    )

    # --------------------------------------------------------
    # Remove the two zero tail bits added by encoder.
    # --------------------------------------------------------

    if terminate:

        if len(decoded_bits) >= MEMORY:
            decoded_bits = decoded_bits[:-MEMORY]

    return "".join(
        str(bit)
        for bit in decoded_bits
    )


# ============================================================
# ASCII CONVERSION
# ============================================================

def bits_to_ascii(
    bit_string: str
) -> str:
    """
    Convert a binary string into ASCII text.

    Example:

        01001000 -> H
    """

    if len(bit_string) % 8 != 0:

        raise ValueError(
            "Bit string length must be a multiple of 8 "
            "for ASCII conversion."
        )

    characters = []

    for index in range(
        0,
        len(bit_string),
        8
    ):

        byte = bit_string[
            index:index + 8
        ]

        characters.append(
            chr(int(byte, 2))
        )

    return "".join(characters)

# ============================================================
# DE-INTERLEAVER TEST
# ============================================================

def test_deinterleaver():
    """
    Verify that de-interleaving correctly reverses
    the block interleaver used by the generator.
    """

    original_bits = (
        "1011001010101011"
        "1110001110101010"
        "0101010111110000"
    )

    # --------------------------------------------------------
    # Local copy of the generator's interleaver
    # --------------------------------------------------------

    cols = 8

    pad_len = (
        cols - len(original_bits) % cols
    ) % cols

    padded_bits = (
        original_bits
        + "0" * pad_len
    )

    rows = len(padded_bits) // cols

    matrix = []

    for row in range(rows):

        matrix.append(
            [
                int(bit)
                for bit in padded_bits[
                    row * cols:
                    (row + 1) * cols
                ]
            ]
        )

    # Transpose exactly as the generator does.
    interleaved_matrix = list(
        zip(*matrix)
    )

    interleaved_bits = "".join(
        str(bit)
        for row in interleaved_matrix
        for bit in row
    )

    # --------------------------------------------------------
    # De-interleave
    # --------------------------------------------------------

    recovered_bits = deinterleave_bits(
        interleaved_bits,
        cols=cols,
        original_length=len(original_bits)
    )

    print()
    print("=" * 60)
    print("DE-INTERLEAVER TEST")
    print("=" * 60)

    print(
        f"Original bits:    "
        f"{original_bits}"
    )

    print(
        f"Interleaved bits: "
        f"{interleaved_bits}"
    )

    print(
        f"Recovered bits:   "
        f"{recovered_bits}"
    )

    if recovered_bits == original_bits:

        print()
        print("✅ DE-INTERLEAVER TEST PASSED")

    else:

        print()
        print("❌ DE-INTERLEAVER TEST FAILED")

        raise AssertionError(
            "De-interleaver did not recover "
            "the original bits."
        )

# ============================================================
# TEST / DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    test_deinterleaver()

    print("=" * 60)
    print("VITERBI DECODER TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Original message
    # --------------------------------------------------------

    message = "SIH2026"

    print(
        f"Original message: {message}"
    )

    # Convert message to bits.
    original_bits = "".join(
        f"{ord(character):08b}"
        for character in message
    )

    print(
        f"Original bits: {len(original_bits)} bits"
    )

    # --------------------------------------------------------
    # Local convolutional encoder
    #
    # This is used ONLY for testing the decoder.
    # The real generator is dataset_generator.py.
    # --------------------------------------------------------

    state = 0
    encoded = []

    input_bits = (
        original_bits
        + "0" * MEMORY
    )

    for bit_char in input_bits:

        bit = int(bit_char)

        shift_register = (
            state << 1
        ) | bit

        out_1 = parity(
            shift_register & GENERATORS[0]
        )

        out_2 = parity(
            shift_register & GENERATORS[1]
        )

        encoded.extend(
            [out_1, out_2]
        )

        state = (
            ((state << 1) | bit)
            & ((1 << MEMORY) - 1)
        )

    print(
        f"Encoded bits: {len(encoded)} bits"
    )

    # --------------------------------------------------------
    # Introduce artificial bit errors
    #
    # This demonstrates why Viterbi is useful.
    # --------------------------------------------------------

    received = encoded.copy()

    # Flip a few coded bits.
    error_positions = [
        5,
        18,
        37,
        61,
        89
    ]

    for position in error_positions:

        if position < len(received):

            received[position] ^= 1

    print(
        f"Injected errors: "
        f"{sum(
            encoded[i] != received[i]
            for i in range(len(encoded))
        )}"
    )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    decoded_bits = viterbi_decode(
        received,
        terminate=True
    )

    decoded_message = bits_to_ascii(
        decoded_bits
    )

    print(
        f"Decoded message: {decoded_message}"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if decoded_message == message:

        print()
        print("✅ VITERBI TEST PASSED")
        print(
            "The original message survived "
            "the injected bit errors."
        )

    else:

        print()
        print("❌ VITERBI TEST FAILED")
        print(
            "Decoded message does not match "
            "the original message."
        )