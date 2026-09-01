import numpy as np

from ingestion.iq_wav_reader import load_signal_file


result = load_signal_file(
    "sample_data/test_tone.iq",
    sample_rate=2_000_000,
    iq_dtype="float32"
)

signal = result["samples"]

print()
print("========== SIGNAL CHECK ==========")

print("Shape:", signal.shape)
print("Dtype:", signal.dtype)
print("Sample rate:", result["sample_rate"])
print("Duration:", result["duration_sec"])

print()
print("First 10 samples:")
print(signal[:10])

print()
print("Mean after DC removal:")
print(np.mean(signal))

print()
print("Maximum magnitude:")
print(np.max(np.abs(signal)))

print("==================================")
