"""
SIH 2026 - Signal Intelligence Toolkit
Module 6: De-interleaving and Interleaving Engine
Owner: Tirth

Supports:
- Block Interleaving & De-interleaving (Matrix row-write / col-read)
- Convolutional Interleaving & De-interleaving (Forney/Ramsey delay line shift registers)
- Diagonal Interleaving & De-interleaving
- Pseudo-Random (PN) Interleaving & De-interleaving

Conforms to docs/interface_contract.md (Step 4: Demodulation -> De-interleaving -> FEC).
"""

from typing import Union, Optional, Tuple, List
import numpy as np


# ============================================================
# 1. BLOCK INTERLEAVER / DE-INTERLEAVER
# ============================================================

class BlockInterleaver:
    """
    Standard Matrix Block Interleaver & De-interleaver.
    
    In block interleaving:
    - Transmitter writes bits row-by-row into an (R x C) matrix,
      and reads them out column-by-column (transposed).
    - Receiver writes interleaved bits column-by-column into an (R x C) matrix,
      and reads them out row-by-row to reconstruct original sequential order.
      
    This disperses burst errors caused by fading/jamming across multiple
    codewords so FEC decoders can easily correct them.
    """

    def __init__(self, cols: int = 8, pad_value: int = 0):
        if cols <= 0:
            raise ValueError(f"Interleaver column count must be positive, got {cols}")
        self.cols = cols
        self.pad_value = pad_value

    def interleave(
        self,
        bits: Union[str, np.ndarray, List[int]],
        cols: Optional[int] = None
    ) -> Tuple[Union[str, np.ndarray], int]:
        """
        Interleave input bits using matrix transpose.
        
        Parameters
        ----------
        bits : str, np.ndarray, or list of ints
            Sequential bitstream (e.g. "01101..." or np.array([0, 1, 1, 0, 1]))
        cols : int, optional
            Number of matrix columns. Defaults to self.cols (8).
            
        Returns
        -------
        interleaved_bits : str or np.ndarray (same type as input)
        pad_len : int
            Number of padding bits appended to satisfy matrix dimensions.
        """
        c = cols or self.cols
        is_str_input = isinstance(bits, str)
        
        # Convert to numpy array of ints
        if is_str_input:
            arr = np.array([int(b) for b in bits], dtype=np.uint8)
        else:
            arr = np.asarray(bits, dtype=np.uint8)

        orig_len = len(arr)
        pad_len = (c - (orig_len % c)) % c
        if pad_len > 0:
            arr = np.pad(arr, (0, pad_len), constant_values=self.pad_value)

        rows = len(arr) // c
        matrix = arr.reshape((rows, c))
        interleaved = matrix.T.flatten()

        if is_str_input:
            return "".join(map(str, interleaved)), pad_len
        return interleaved, pad_len

    def deinterleave(
        self,
        bits: Union[str, np.ndarray, List[int]],
        cols: Optional[int] = None,
        original_length: Optional[int] = None
    ) -> Union[str, np.ndarray]:
        """
        De-interleave interleaved bits, reversing the matrix transpose.
        
        Parameters
        ----------
        bits : str, np.ndarray, or list of ints
            Interleaved bitstream
        cols : int, optional
            Interleaver columns (must match transmitter). Defaults to self.cols (8).
        original_length : int, optional
            If specified, strips any padding bits added during interleaving.
            
        Returns
        -------
        deinterleaved_bits : str or np.ndarray (same type as input)
        """
        c = cols or self.cols
        is_str_input = isinstance(bits, str)

        if is_str_input:
            if not bits:
                return ""
            arr = np.array([int(b) for b in bits], dtype=np.uint8)
        else:
            arr = np.asarray(bits, dtype=np.uint8)
            if len(arr) == 0:
                return np.array([], dtype=np.uint8)

        if len(arr) % c != 0:
            # If length is not perfectly divisible, trim to multiple of c or pad
            valid_len = (len(arr) // c) * c
            arr = arr[:valid_len]

        rows = len(arr) // c
        # At receiver: received stream was read out col-by-col (c columns, each of length rows)
        # So we reshape into (c, rows) and transpose back to (rows, c)
        matrix = arr.reshape((c, rows))
        deinterleaved = matrix.T.flatten()

        if original_length is not None and original_length < len(deinterleaved):
            deinterleaved = deinterleaved[:original_length]

        if is_str_input:
            return "".join(map(str, deinterleaved))
        return deinterleaved


# ============================================================
# 2. CONVOLUTIONAL INTERLEAVER / DE-INTERLEAVER
# ============================================================

class ConvolutionalInterleaver:
    """
    Forney-type Convolutional Interleaver / De-interleaver.
    
    Consists of M branches. Branch j has a FIFO shift register with delay (j * D).
    The de-interleaver has branch j with delay ((M - 1 - j) * D).
    Total end-to-end delay through interleaver + de-interleaver is (M - 1) * D.
    """

    def __init__(self, branches: int = 4, delay_step: int = 2):
        self.M = branches
        self.D = delay_step

    def interleave(self, bits: Union[str, np.ndarray]) -> Union[str, np.ndarray]:
        is_str = isinstance(bits, str)
        bit_list = [int(b) for b in bits] if is_str else list(bits)
        
        fifos = [[0] * (j * self.D) for j in range(self.M)]
        output = []

        for idx, bit in enumerate(bit_list):
            branch = idx % self.M
            if self.D == 0 or branch == 0:
                output.append(bit)
            else:
                fifos[branch].append(bit)
                output.append(fifos[branch].pop(0))

        if is_str:
            return "".join(map(str, output))
        return np.array(output, dtype=np.uint8)

    def deinterleave(self, bits: Union[str, np.ndarray]) -> Union[str, np.ndarray]:
        is_str = isinstance(bits, str)
        bit_list = [int(b) for b in bits] if is_str else list(bits)

        # De-interleaver complementary delays: branch j has ((M - 1 - j) * D)
        fifos = [[0] * ((self.M - 1 - j) * self.D) for j in range(self.M)]
        output = []

        for idx, bit in enumerate(bit_list):
            branch = idx % self.M
            delay = (self.M - 1 - branch) * self.D
            if delay == 0:
                output.append(bit)
            else:
                fifos[branch].append(bit)
                output.append(fifos[branch].pop(0))

        # Discard transient initialization delay (M * (M - 1) * D) if desired
        total_delay = self.M * (self.M - 1) * self.D
        if len(output) > total_delay:
            recovered = output[total_delay:]
        else:
            recovered = output

        if is_str:
            return "".join(map(str, recovered))
        return np.array(recovered, dtype=np.uint8)


# ============================================================
# 3. DIAGONAL INTERLEAVER / DE-INTERLEAVER
# ============================================================

class DiagonalInterleaver:
    """
    Diagonal Interleaver:
    Writes bits into an (N x N) block and reads out diagonally with cyclic wrap.
    """

    def __init__(self, size: int = 8):
        self.size = size

    def interleave(self, bits: Union[str, np.ndarray]) -> Union[str, np.ndarray]:
        is_str = isinstance(bits, str)
        arr = np.array([int(b) for b in bits], dtype=np.uint8) if is_str else np.asarray(bits, dtype=np.uint8)
        
        block_len = self.size * self.size
        pad_len = (block_len - (len(arr) % block_len)) % block_len
        if pad_len > 0:
            arr = np.pad(arr, (0, pad_len), constant_values=0)

        out_blocks = []
        for blk_idx in range(0, len(arr), block_len):
            block = arr[blk_idx : blk_idx + block_len].reshape((self.size, self.size))
            interleaved_blk = np.zeros_like(block)
            for r in range(self.size):
                for c in range(self.size):
                    interleaved_blk[(r + c) % self.size, c] = block[r, c]
            out_blocks.extend(interleaved_blk.flatten().tolist())

        if is_str:
            return "".join(map(str, out_blocks))
        return np.array(out_blocks, dtype=np.uint8)

    def deinterleave(self, bits: Union[str, np.ndarray], original_length: Optional[int] = None) -> Union[str, np.ndarray]:
        is_str = isinstance(bits, str)
        arr = np.array([int(b) for b in bits], dtype=np.uint8) if is_str else np.asarray(bits, dtype=np.uint8)
        
        block_len = self.size * self.size
        valid_len = (len(arr) // block_len) * block_len
        arr = arr[:valid_len]

        out_blocks = []
        for blk_idx in range(0, len(arr), block_len):
            interleaved_blk = arr[blk_idx : blk_idx + block_len].reshape((self.size, self.size))
            orig_blk = np.zeros_like(interleaved_blk)
            for r in range(self.size):
                for c in range(self.size):
                    orig_blk[r, c] = interleaved_blk[(r + c) % self.size, c]
            out_blocks.extend(orig_blk.flatten().tolist())

        if original_length is not None and original_length < len(out_blocks):
            out_blocks = out_blocks[:original_length]

        if is_str:
            return "".join(map(str, out_blocks))
        return np.array(out_blocks, dtype=np.uint8)


# ============================================================
# 4. PSEUDO-RANDOM (PN) INTERLEAVER
# ============================================================

class PseudoRandomInterleaver:
    """
    Pseudo-Random Permutation Interleaver:
    Permutes bits within fixed blocks based on a deterministic PRNG seed.
    """

    def __init__(self, block_size: int = 64, seed: int = 42):
        self.block_size = block_size
        self.seed = seed
        rng = np.random.RandomState(seed)
        self.perm = rng.permutation(block_size)
        self.inv_perm = np.argsort(self.perm)

    def interleave(self, bits: Union[str, np.ndarray]) -> Union[str, np.ndarray]:
        is_str = isinstance(bits, str)
        arr = np.array([int(b) for b in bits], dtype=np.uint8) if is_str else np.asarray(bits, dtype=np.uint8)
        
        pad_len = (self.block_size - (len(arr) % self.block_size)) % self.block_size
        if pad_len > 0:
            arr = np.pad(arr, (0, pad_len), constant_values=0)

        out = []
        for i in range(0, len(arr), self.block_size):
            chunk = arr[i : i + self.block_size]
            out.extend(chunk[self.perm].tolist())

        if is_str:
            return "".join(map(str, out))
        return np.array(out, dtype=np.uint8)

    def deinterleave(self, bits: Union[str, np.ndarray], original_length: Optional[int] = None) -> Union[str, np.ndarray]:
        is_str = isinstance(bits, str)
        arr = np.array([int(b) for b in bits], dtype=np.uint8) if is_str else np.asarray(bits, dtype=np.uint8)

        valid_len = (len(arr) // self.block_size) * self.block_size
        arr = arr[:valid_len]

        out = []
        for i in range(0, len(arr), self.block_size):
            chunk = arr[i : i + self.block_size]
            out.extend(chunk[self.inv_perm].tolist())

        if original_length is not None and original_length < len(out):
            out = out[:original_length]

        if is_str:
            return "".join(map(str, out))
        return np.array(out, dtype=np.uint8)


# ============================================================
# 5. BACKWARD-COMPATIBLE UTILITY FUNCTIONS
# ============================================================

def interleave(bits: Union[str, np.ndarray, List[int]], cols: int = 8) -> Union[str, np.ndarray]:
    """
    Standard block interleaver: arranges bits into a matrix with `cols` columns
    and transposes them (row-write, col-read).
    """
    interleaver = BlockInterleaver(cols=cols)
    out, _ = interleaver.interleave(bits, cols=cols)
    return out


def deinterleave(
    bits: Union[str, np.ndarray, List[int]],
    cols: int = 8,
    original_length: Optional[int] = None
) -> Union[str, np.ndarray]:
    """
    Standard block de-interleaver: reverses the matrix transpose (col-write, row-read).
    """
    interleaver = BlockInterleaver(cols=cols)
    return interleaver.deinterleave(bits, cols=cols, original_length=original_length)


def interleave_bits(bit_str: str, cols: int = 8) -> str:
    """
    Standard block interleaver used across the repository and dataset generator.
    """
    return str(interleave(bit_str, cols=cols))


def deinterleave_bits(
    bit_str: str,
    cols: int = 8,
    original_length: Optional[int] = None
) -> str:
    """
    Standard block de-interleaver reversing interleave_bits.
    """
    return str(deinterleave(bit_str, cols=cols, original_length=original_length))


# ============================================================
# 6. PIPELINE INTERFACE CONTRACT (Step 4)
# ============================================================

def process_deinterleaving(
    stage_data: dict,
    interleaver_type: str = "block",
    cols: int = 8,
    original_length: Optional[int] = None
) -> dict:
    """
    Interface function fulfilling docs/interface_contract.md Step 4:
    Demodulation -> De-interleaving -> FEC Decoding.
    
    Parameters
    ----------
    stage_data : dict
        Dict from demodulator containing 'bitstream' (str or np.ndarray).
    interleaver_type : str
        One of "block", "convolutional", "diagonal", "pseudo_random", or "none".
    cols : int
        Interleaver columns (for block interleaver).
        
    Returns
    -------
    dict
        Extends stage_data with:
        'deinterleaved_bits': np.ndarray (uint8)
        'interleaver_type': str
    """
    result = dict(stage_data)
    bitstream = stage_data.get("bitstream", "")

    # Normalize to string for internal processing
    if isinstance(bitstream, np.ndarray):
        bit_str = "".join(map(str, bitstream.astype(int)))
    else:
        bit_str = str(bitstream)

    itype = interleaver_type.lower().strip()
    if itype == "block":
        deint_str = deinterleave_bits(bit_str, cols=cols, original_length=original_length)
    elif itype == "convolutional":
        conv_deint = ConvolutionalInterleaver(branches=4, delay_step=2)
        deint_str = conv_deint.deinterleave(bit_str)
    elif itype == "diagonal":
        diag_deint = DiagonalInterleaver(size=cols)
        deint_str = diag_deint.deinterleave(bit_str, original_length=original_length)
    elif itype in ("pseudo_random", "pn"):
        pn_deint = PseudoRandomInterleaver(block_size=64, seed=42)
        deint_str = pn_deint.deinterleave(bit_str, original_length=original_length)
    elif itype == "none":
        deint_str = bit_str
    else:
        raise ValueError(f"Unsupported interleaver_type: {interleaver_type}")

    # Produce 1D np.ndarray uint8 as required by contract
    deint_arr = np.array([int(b) for b in deint_str], dtype=np.uint8) if deint_str else np.array([], dtype=np.uint8)

    result["deinterleaved_bits"] = deint_arr
    result["deinterleaved_bitstring"] = deint_str
    result["interleaver_type"] = itype
    return result
