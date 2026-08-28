"""
Signal ingestion module.

Reads .WAV and raw .IQ files and converts both into a common
complex64 NumPy representation.

Pipeline:
    File -> Reader -> I/Q -> Complex -> DC removal -> Normalization

Interface contract:
    load_signal_file(...) returns a dictionary containing at least:
        samples
        sample_rate
        source_format
        duration_sec
"""

from pathlib import Path
import numpy as np
import soundfile as sf


# ============================================================
# WAV READER
# ============================================================

def read_wav(filepath: str) -> dict:
    """
    Read a WAV file and convert it to complex64 samples.

    Stereo WAV:
        Channel 0 -> I
        Channel 1 -> Q

    Mono WAV:
        Signal -> I
        Q = 0
    """

    filepath = str(filepath)

    # Read audio samples
    data, sample_rate = sf.read(
        filepath,
        dtype="float32",
        always_2d=False
    )

    # Get file information
    info = sf.info(filepath)

    # --------------------------------------------------------
    # Convert WAV data to complex I/Q
    # --------------------------------------------------------

    if data.ndim == 2 and data.shape[1] >= 2:
        # Stereo WAV
        i_samples = data[:, 0]
        q_samples = data[:, 1]

        samples = i_samples + 1j * q_samples

        channels = data.shape[1]

    else:
        # Mono WAV
        samples = data.astype(np.complex64)
        channels = 1

    samples = samples.astype(np.complex64)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    num_samples = len(samples)

    duration_sec = (
        num_samples / sample_rate
        if sample_rate > 0
        else 0.0
    )

    # Estimate bit depth from subtype
    bit_depth = info.subtype

    return {
        "samples": samples,
        "sample_rate": float(sample_rate),
        "source_format": "wav",
        "duration_sec": float(duration_sec),

        # Additional metadata
        "num_samples": int(num_samples),
        "channels": int(channels),
        "bit_depth": bit_depth,
        "file_name": Path(filepath).name,
    }


# ============================================================
# IQ READER
# ============================================================

def read_iq(
    filepath: str,
    sample_rate: float,
    dtype: str = "float32"
) -> dict:
    """
    Read a raw IQ file.

    Assumes interleaved format:

        I Q I Q I Q I Q ...

    Supported input formats:
        float32
        int16
        uint8

    Raw IQ files have no header, therefore sample_rate
    and dtype must be supplied externally.
    """

    filepath = str(filepath)

    if sample_rate is None or sample_rate <= 0:
        raise ValueError(
            "A positive sample_rate is required for raw .iq files."
        )

    supported_dtypes = {
        "float32": np.float32,
        "int16": np.int16,
        "uint8": np.uint8,
    }

    if dtype not in supported_dtypes:
        raise ValueError(
            f"Unsupported IQ dtype '{dtype}'. "
            f"Supported: {list(supported_dtypes.keys())}"
        )

    # --------------------------------------------------------
    # Read raw bytes
    # --------------------------------------------------------

    raw = np.fromfile(
        filepath,
        dtype=supported_dtypes[dtype]
    )

    if raw.size == 0:
        raise ValueError("IQ file is empty.")

    # --------------------------------------------------------
    # Convert raw values to float32
    # --------------------------------------------------------

    if dtype == "float32":

        raw = raw.astype(np.float32)

    elif dtype == "int16":

        raw = raw.astype(np.float32)

        # Convert approximately to [-1, 1]
        raw /= 32768.0

    elif dtype == "uint8":

        raw = raw.astype(np.float32)

        # RTL-SDR style unsigned samples
        raw = (raw - 127.5) / 127.5

    # --------------------------------------------------------
    # Validate I/Q pairing
    # --------------------------------------------------------

    if raw.size % 2 != 0:
        raise ValueError(
            "IQ file contains an odd number of values. "
            "Expected interleaved I,Q pairs."
        )

    # --------------------------------------------------------
    # Split I and Q
    # --------------------------------------------------------

    i_samples = raw[0::2]
    q_samples = raw[1::2]

    # --------------------------------------------------------
    # Build complex signal
    # --------------------------------------------------------

    samples = (
        i_samples + 1j * q_samples
    ).astype(np.complex64)

    num_samples = len(samples)

    duration_sec = num_samples / sample_rate

    return {
        "samples": samples,
        "sample_rate": float(sample_rate),
        "source_format": "iq",
        "duration_sec": float(duration_sec),

        # Additional metadata
        "num_samples": int(num_samples),
        "channels": 2,
        "bit_depth": dtype,
        "iq_dtype": dtype,
        "file_name": Path(filepath).name,
    }


# ============================================================
# PREPROCESSING
# ============================================================

def remove_dc(signal: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from a complex signal.

    DC removal:
        x_clean = x - mean(x)
    """

    signal = np.asarray(
        signal,
        dtype=np.complex64
    )

    if signal.size == 0:
        return signal

    dc_offset = np.mean(signal)

    return (
        signal - dc_offset
    ).astype(np.complex64)


def normalize_signal(
    signal: np.ndarray,
    method: str = "peak"
) -> np.ndarray:
    """
    Normalize signal amplitude.

    method='peak':
        Maximum magnitude becomes 1.

    Returns complex64 signal.
    """

    signal = np.asarray(
        signal,
        dtype=np.complex64
    )

    if signal.size == 0:
        return signal

    if method == "peak":

        peak = np.max(np.abs(signal))

        if peak == 0:
            return signal

        signal = signal / peak

    else:
        raise ValueError(
            f"Unsupported normalization method: {method}"
        )

    return signal.astype(np.complex64)


def preprocess_signal(
    signal: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Complete Level-1 preprocessing pipeline.

    1. DC removal
    2. Normalization
    """

    signal = remove_dc(signal)

    if normalize:
        signal = normalize_signal(signal)

    return signal.astype(np.complex64)


# ============================================================
# COMMON ENTRY POINT
# ============================================================

def load_signal_file(
    filepath: str,
    sample_rate: float = None,
    iq_dtype: str = "float32",
    preprocess: bool = True
) -> dict:
    """
    Main ingestion entry point.

    Automatically selects the correct reader based on
    file extension.

    WAV:
        sample rate comes from the WAV header.

    IQ:
        sample rate and dtype must be supplied externally.

    Returns a dictionary containing the processed signal
    and metadata required by downstream modules.
    """

    filepath = str(filepath)

    if not Path(filepath).exists():
        raise FileNotFoundError(
            f"File not found: {filepath}"
        )

    extension = Path(filepath).suffix.lower()

    # --------------------------------------------------------
    # Detect file type
    # --------------------------------------------------------

    if extension == ".wav":

        result = read_wav(filepath)

    elif extension == ".iq":

        result = read_iq(
            filepath,
            sample_rate=sample_rate,
            dtype=iq_dtype
        )

    else:

        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Supported formats are .iq and .wav."
        )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    if preprocess:

        result["samples"] = preprocess_signal(
            result["samples"]
        )

    # --------------------------------------------------------
    # Recalculate duration from final sample count
    # --------------------------------------------------------

    if result["sample_rate"] > 0:

        result["duration_sec"] = (
            len(result["samples"])
            / result["sample_rate"]
        )

    return result


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python iq_wav_reader.py <file> [sample_rate] [iq_dtype]"
        )

        sys.exit(1)

    filepath = sys.argv[1]

    sample_rate = (
        float(sys.argv[2])
        if len(sys.argv) >= 3
        else None
    )

    iq_dtype = (
        sys.argv[3]
        if len(sys.argv) >= 4
        else "float32"
    )

    try:

        result = load_signal_file(
            filepath,
            sample_rate=sample_rate,
            iq_dtype=iq_dtype
        )

        print("\n========== SIGNAL LOADED ==========")

        print(
            f"File          : "
            f"{result['file_name']}"
        )

        print(
            f"Format        : "
            f"{result['source_format']}"
        )

        print(
            f"Samples       : "
            f"{result['num_samples']}"
        )

        print(
            f"Sample rate   : "
            f"{result['sample_rate']} Hz"
        )

        print(
            f"Duration      : "
            f"{result['duration_sec']:.6f} sec"
        )

        print(
            f"Array dtype   : "
            f"{result['samples'].dtype}"
        )

        print(
            f"Maximum amp   : "
            f"{np.max(np.abs(result['samples'])):.4f}"
        )

        print("===================================\n")

    except Exception as exc:

        print(f"ERROR: {exc}")
        sys.exit(1)