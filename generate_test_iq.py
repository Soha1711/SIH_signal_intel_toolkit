import numpy as np
from pathlib import Path

# Output file
output_file = Path("sample_data/test_tone.iq")

# Signal parameters
sample_rate = 2_000_000
duration = 0.02
frequency = 100_000
amplitude = 1.0

# Number of complex samples
num_samples = int(sample_rate * duration)

# Time
t = np.arange(num_samples) / sample_rate

# Generate complex signal
signal = amplitude * np.exp(2j * np.pi * frequency * t)

# Create interleaved float32 IQ data
iq_data = np.empty(2 * num_samples, dtype=np.float32)

iq_data[0::2] = signal.real
iq_data[1::2] = signal.imag

# Make sure sample_data exists
output_file.parent.mkdir(parents=True, exist_ok=True)

# Save IQ file
iq_data.tofile(output_file)

print("=" * 60)
print("TEST IQ FILE GENERATED")
print("=" * 60)
print(f"File            : {output_file}")
print(f"Sample rate     : {sample_rate} Hz")
print(f"Frequency       : {frequency} Hz")
print(f"Duration        : {duration} sec")
print(f"Complex samples : {num_samples}")
print(f"File size       : {output_file.stat().st_size} bytes")
print("=" * 60)