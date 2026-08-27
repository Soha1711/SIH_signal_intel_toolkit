"""
File Ingestion module — reads .iq and .wav files and normalises both into a
common complex-sample array format, per docs/interface_contract.md (Step 1).

Owner: Soha
"""

import numpy as np
import soundfile as sf


def read_wav(filepath: str) -> dict:
    """
    Reads a .wav file. If stereo, treats channel 0 as I and channel 1 as Q
    (common convention for SDR recordings saved as stereo WAV).
    If mono, returns real-valued samples cast to complex (Q = 0).
    """
    data, sample_rate = sf.read(filepath, dtype="float32")

    if data.ndim == 2 and data.shape[1] >= 2:
        samples = data[:, 0] + 1j * data[:, 1]
    else:
        samples = data.astype(np.complex64)

    samples = samples.astype(np.complex64)
    duration_sec = len(samples) / sample_rate

    return {
        "samples": samples,
        "sample_rate": float(sample_rate),
        "source_format": "wav",
        "duration_sec": duration_sec,
    }


def read_iq(filepath: str, sample_rate: float, dtype: str = "float32") -> dict:
    """
    Reads a raw .iq file. Raw IQ files have NO header — sample rate and dtype
    must be known or supplied by the user via the GUI, since the file itself
    doesn't declare them.

    Assumes interleaved I,Q,I,Q,... format, which is the most common raw IQ
    layout (e.g. GNU Radio's default file sink format).

    dtype options: "float32" (GNU Radio default), "int16", "uint8" (RTL-SDR default)
    """
    raw = np.fromfile(filepath, dtype=dtype)

    # Normalise integer formats to [-1, 1] range like float32 IQ
    if dtype == "int16":
        raw = raw.astype(np.float32) / 32768.0
    elif dtype == "uint8":
        raw = (raw.astype(np.float32) - 127.5) / 127.5

    if len(raw) % 2 != 0:
        raw = raw[:-1]  # drop a trailing odd sample, shouldn't normally happen

    i_samples = raw[0::2]
    q_samples = raw[1::2]
    samples = (i_samples + 1j * q_samples).astype(np.complex64)

    duration_sec = len(samples) / sample_rate

    return {
        "samples": samples,
        "sample_rate": float(sample_rate),
        "source_format": "iq",
        "duration_sec": duration_sec,
    }


def load_signal_file(filepath: str, sample_rate: float = None, iq_dtype: str = "float32") -> dict:
    """
    Convenience entry point — picks the right reader based on file extension.
    sample_rate and iq_dtype are required for .iq files since raw IQ has no header.
    """
    if filepath.lower().endswith(".wav"):
        return read_wav(filepath)
    elif filepath.lower().endswith(".iq"):
        if sample_rate is None:
            raise ValueError("sample_rate must be provided for .iq files (no header to read it from)")
        return read_iq(filepath, sample_rate=sample_rate, dtype=iq_dtype)
    else:
        raise ValueError(f"Unsupported file extension for: {filepath}")


if __name__ == "__main__":
    # Quick manual test — replace with a real sample file once sample_data/ has one
    import sys
    if len(sys.argv) < 2:
        print("Usage: python iq_wav_reader.py <path_to_file> [sample_rate_if_iq]")
        sys.exit(1)

    path = sys.argv[1]
    sr = float(sys.argv[2]) if len(sys.argv) > 2 else None
    result = load_signal_file(path, sample_rate=sr)
    print(f"Loaded {result['source_format']} file:")
    print(f"  samples: {result['samples'].shape}, dtype {result['samples'].dtype}")
    print(f"  sample_rate: {result['sample_rate']} Hz")
    print(f"  duration: {result['duration_sec']:.3f} sec")
