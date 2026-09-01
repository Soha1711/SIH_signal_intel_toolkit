import numpy as np

from param_estimation.waterfall_constellation import (
    compute_waterfall,
    compute_constellation,
    analyze_waterfall_constellation
)


def test_waterfall_output():

    sample_rate = 100000

    signal = np.exp(
        1j * 2 * np.pi * 10000
        * np.arange(10000)
        / sample_rate
    )

    frequencies, times, power_db = (
        compute_waterfall(
            signal,
            sample_rate
        )
    )

    assert len(frequencies) > 0

    assert len(times) > 0

    assert power_db.shape == (
        len(frequencies),
        len(times)
    )


def test_constellation_output():

    signal = np.array(
        [
            1 + 1j,
            1 - 1j,
            -1 + 1j,
            -1 - 1j
        ],
        dtype=np.complex64
    )

    i_data, q_data = (
        compute_constellation(signal)
    )

    assert len(i_data) == 4

    assert len(q_data) == 4

    assert np.array_equal(
        i_data,
        [1, 1, -1, -1]
    )

    assert np.array_equal(
        q_data,
        [1, -1, 1, -1]
    )


def test_complete_analysis():

    sample_rate = 100000

    signal = np.exp(
        1j * 2 * np.pi * 5000
        * np.arange(5000)
        / sample_rate
    ).astype(
        np.complex64
    )

    result = analyze_waterfall_constellation(
        signal,
        sample_rate
    )

    assert "waterfall_frequency" in result

    assert "waterfall_time" in result

    assert "waterfall_power_db" in result

    assert "constellation_i" in result

    assert "constellation_q" in result
