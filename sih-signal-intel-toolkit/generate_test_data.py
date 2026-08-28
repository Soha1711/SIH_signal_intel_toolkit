import numpy as np
import soundfile as sf
from pathlib import Path


# ---------------------------------------------------------
# Output folder
# ---------------------------------------------------------

OUTPUT_DIR = Path("sample_data")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# SIGNAL PARAMETERS
# ---------------------------------------------------------

SAMPLE_RATE = 2_000_000       # 2 MHz
FREQUENCY = 100_000          # 100 kHz
DURATION = 1.0               # 1 second

NUM_SAMPLES = int(SAMPLE_RATE * DURATION)


# ---------------------------------------------------------
# Generate time axis
# ---------------------------------------------------------

t = np.arange(NUM_SAMPLES) / SAMPLE_RATE


# ---------------------------------------------------------
# Generate complex IQ signal
# ---------------------------------------------------------

signal = np.exp(
    1j * 2 * np.pi * FREQUENCY * t
).astype(np.complex64)


# =========================================================
# 1. CREATE RAW IQ FILE
# =========================================================

# Create interleaved I/Q data:
#
# I1 Q1 I2 Q2 I3 Q3 ...
#

iq_data = np.empty(
    signal.size * 2,
    dtype=np.float32
)

iq_data[0::2] = signal.real
iq_data[1::2] = signal.imag


iq_path = OUTPUT_DIR / "test_tone.iq"

iq_data.tofile(iq_path)


# =========================================================
# 2. CREATE WAV FILE
# =========================================================

# Stereo WAV:
#
# Channel 0 = I
# Channel 1 = Q
#

wav_data = np.column_stack(
    (
        signal.real,
        signal.imag
    )
).astype(np.float32)


wav_path = OUTPUT_DIR / "test_signal.wav"

sf.write(
    wav_path,
    wav_data,
    SAMPLE_RATE
)


# =========================================================
# PRINT INFORMATION
# =========================================================

print()
print("=" * 50)
print("TEST SIGNALS GENERATED")
print("=" * 50)

print(f"IQ file      : {iq_path}")
print(f"WAV file     : {wav_path}")
print(f"Sample rate  : {SAMPLE_RATE} Hz")
print(f"Frequency    : {FREQUENCY} Hz")
print(f"Duration     : {DURATION} sec")
print(f"Samples      : {NUM_SAMPLES}")

print("=" * 50)
print()