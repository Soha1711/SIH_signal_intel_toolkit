import numpy as np
import soundfile as sf

from ingestion.iq_wav_reader import (
    read_iq,
    read_wav,
    remove_dc,
    normalize_signal,
    preprocess_signal,
    load_signal_file,
)


# ============================================================
# TEST 1 — IQ READING
# ============================================================

def test_iq_reader(tmp_path):

    filepath = tmp_path / "test.iq"

    # Create:
    # I1 Q1 I2 Q2

    raw = np.array(
        [1, 2, 3, 4],
        dtype=np.float32
    )

    raw.tofile(filepath)

    result = read_iq(
        str(filepath),
        sample_rate=1000,
        dtype="float32"
    )

    # Check file type
    assert result["source_format"] == "iq"

    # Check sample rate
    assert result["sample_rate"] == 1000

    # 4 raw values = 2 complex samples
    assert result["num_samples"] == 2

    # Check datatype
    assert result["samples"].dtype == np.complex64

    # Check actual I/Q conversion
    expected = np.array(
        [
            1 + 2j,
            3 + 4j
        ],
        dtype=np.complex64
    )

    assert np.allclose(
        result["samples"],
        expected
    )


# ============================================================
# TEST 2 — WAV READING
# ============================================================

def test_wav_reader(tmp_path):

    filepath = tmp_path / "test.wav"

    sample_rate = 8000

    # Create I and Q channels
    i = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float32
    )

    q = np.array(
        [0.4, 0.5, 0.6],
        dtype=np.float32
    )

    stereo = np.column_stack(
        (i, q)
    )

    # Save WAV
    sf.write(
        filepath,
        stereo,
        sample_rate
    )

    result = read_wav(str(filepath))

    # Check format
    assert result["source_format"] == "wav"

    # Check sample rate
    assert result["sample_rate"] == sample_rate

    # Check channels
    assert result["channels"] == 2

    # Check complex representation
    assert result["samples"].dtype == np.complex64

    # Check number of samples
    assert result["num_samples"] == 3


# ============================================================
# TEST 3 — DC REMOVAL
# ============================================================

def test_remove_dc():

    signal = np.array(
        [
            1 + 0j,
            2 + 0j,
            3 + 0j
        ],
        dtype=np.complex64
    )

    cleaned = remove_dc(signal)

    # Mean should become approximately zero
    assert np.isclose(
        np.mean(cleaned),
        0,
        atol=1e-6
    )


# ============================================================
# TEST 4 — NORMALIZATION
# ============================================================

def test_normalization():

    signal = np.array(
        [
            1 + 0j,
            2 + 0j,
            4 + 0j
        ],
        dtype=np.complex64
    )

    normalized = normalize_signal(signal)

    # Maximum magnitude should become 1
    assert np.isclose(
        np.max(np.abs(normalized)),
        1.0,
        atol=1e-6
    )


# ============================================================
# TEST 5 — COMPLETE PREPROCESSING
# ============================================================

def test_preprocess_signal():

    signal = np.array(
        [
            2 + 1j,
            3 + 1j,
            4 + 1j
        ],
        dtype=np.complex64
    )

    processed = preprocess_signal(signal)

    # Output must remain complex64
    assert processed.dtype == np.complex64

    # DC should be approximately zero
    assert np.isclose(
        np.mean(processed),
        0,
        atol=1e-6
    )

    # Signal should be normalized
    assert np.isclose(
        np.max(np.abs(processed)),
        1.0,
        atol=1e-6
    )


# ============================================================
# TEST 6 — COMMON LOAD FUNCTION
# ============================================================

def test_load_signal_file(tmp_path):

    filepath = tmp_path / "test.iq"

    raw = np.array(
        [
            1, 0,
            1, 0
        ],
        dtype=np.float32
    )

    raw.tofile(filepath)

    result = load_signal_file(
        str(filepath),
        sample_rate=1000,
        iq_dtype="float32"
    )

    # Required interface fields
    assert "samples" in result
    assert "sample_rate" in result
    assert "source_format" in result
    assert "duration_sec" in result

    # Check values
    assert result["samples"].dtype == np.complex64
    assert result["sample_rate"] == 1000
    assert result["source_format"] == "iq"


# ============================================================
# TEST 7 — IQ REQUIRES SAMPLE RATE
# ============================================================

def test_iq_requires_sample_rate(tmp_path):

    filepath = tmp_path / "test.iq"

    raw = np.array(
        [
            1, 2,
            3, 4
        ],
        dtype=np.float32
    )

    raw.tofile(filepath)

    # Raw IQ has no header, therefore sample rate
    # must be supplied.

    try:

        load_signal_file(
            str(filepath)
        )

        # If the function does not raise an error,
        # the test should fail.
        assert False, "Expected ValueError"

    except ValueError as exc:

        assert "sample_rate" in str(exc)


# ============================================================
# TEST 8 — INVALID FILE TYPE
# ============================================================

def test_invalid_file_type(tmp_path):

    filepath = tmp_path / "test.txt"

    filepath.write_text(
        "This is not a signal file."
    )

    try:

        load_signal_file(
            str(filepath)
        )

        assert False, "Expected ValueError"

    except ValueError as exc:

        assert "Unsupported file type" in str(exc)


# ============================================================
# TEST 9 — ODD NUMBER OF IQ VALUES
# ============================================================

def test_odd_iq_values(tmp_path):

    filepath = tmp_path / "test.iq"

    # Invalid:
    # I Q I
    #
    # There is no Q for the final I.

    raw = np.array(
        [1, 2, 3],
        dtype=np.float32
    )

    raw.tofile(filepath)

    try:

        read_iq(
            str(filepath),
            sample_rate=1000,
            dtype="float32"
        )

        assert False, "Expected ValueError"

    except ValueError as exc:

        assert "odd number" in str(exc)


# ============================================================
# TEST 10 — FILE DOES NOT EXIST
# ============================================================

def test_file_not_found():

    try:

        load_signal_file(
            "does_not_exist.iq",
            sample_rate=1000
        )

        assert False, "Expected FileNotFoundError"

    except FileNotFoundError:
        pass
def test_real_iq_file_pipeline():

    result = load_signal_file(
        "sample_data/test_tone.iq",
        sample_rate=2_000_000,
        iq_dtype="float32"
    )

    signal = result["samples"]

    assert signal is not None

    assert signal.dtype == np.complex64

    assert len(signal) == 2_000_000

    assert result["sample_rate"] == 2_000_000

    assert result["duration_sec"] == 1.0

    assert result["source_format"] == "iq"

    # DC should be approximately removed
    assert np.abs(np.mean(signal)) < 1e-5

    # Normalized signal should have maximum magnitude ~1
    assert np.isclose(
        np.max(np.abs(signal)),
        1.0,
        atol=1e-5
    )
def test_real_wav_file_pipeline():

    result = load_signal_file(
        "sample_data/test_signal.wav"
    )

    signal = result["samples"]

    assert signal is not None

    assert signal.dtype == np.complex64

    assert len(signal) == 2_000_000

    assert result["sample_rate"] == 2_000_000

    assert result["duration_sec"] == 1.0

    assert result["source_format"] == "wav"

    # DC should be approximately removed
    assert np.abs(np.mean(signal)) < 1e-5

    # Normalized signal
    assert np.isclose(
        np.max(np.abs(signal)),
        1.0,
        atol=1e-5
    )
test_real_iq_file_pipeline()
test_real_wav_file_pipeline()
