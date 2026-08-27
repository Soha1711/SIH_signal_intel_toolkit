# Module Interface Contract

This defines exactly what data each module receives and must return, so everyone
can build and test their module independently without waiting on someone else's
finished code. **If you need to change your module's output shape, flag it in
team chat first** — it likely affects the next module downstream.

---

## 1. File Ingestion → Parameter Estimation
**Owner: Soha**

Input: path to a `.iq` or `.wav` file.

Output: a Python dict —
```python
{
    "samples": np.ndarray,       # complex64 array, shape (N,) — I/Q samples
    "sample_rate": float,        # Hz, from file metadata or user input if unknown
    "source_format": str,        # "iq" or "wav"
    "duration_sec": float
}
```

## 2. Parameter Estimation → Demodulation
**Owner: Soha → Honey**

Output: extends the dict above with —
```python
{
    ...,  # everything from step 1
    "modulation_type": str,      # one of: "FSK", "PSK", "QAM", "ASK"
    "symbol_rate": float,        # Hz
    "occupied_bandwidth": float, # Hz
    "confidence": float          # 0-1, classifier confidence
}
```

## 3. Demodulation → De-interleaving
**Owner: Honey → Tirth**

Output:
```python
{
    "bitstream": np.ndarray,     # 1D array of 0/1 (uint8)
    "symbol_stream": np.ndarray, # raw recovered symbols pre-bit-mapping (for constellation plots)
    "modulation_type": str,      # passed through
}
```

## 4. De-interleaving → FEC Decoding
**Owner: Tirth (internal handoff within own module)**

Output:
```python
{
    "deinterleaved_bits": np.ndarray,  # 1D array of 0/1
    "interleaver_type": str            # "block" | "convolutional" | "diagonal" | "pseudo_random"
}
```

## 5. FEC Decoding → Bitstream Correlation
**Owner: Tirth → Hrutu & Rahul**

Output:
```python
{
    "decoded_bits": np.ndarray,   # error-corrected 1D bit array
    "fec_scheme": str,            # "viterbi" | "reed_solomon" | "concatenated" | "ldpc"
    "error_count": int            # errors corrected, if available from the decoder
}
```

## 6. Bitstream Correlation → GUI Output
**Owner: Hrutu & Rahul**

Output:
```python
{
    "header_start": int,       # bit index
    "header_end": int,
    "payload_start": int,
    "payload_end": int,
    "sync_word_matched": str   # the pattern that was matched, or "" if none found
}
```

---

## Plotting data (available to GUI at any stage)

Any module producing a plot should return plain NumPy arrays, not matplotlib
figures directly, so the GUI layer controls rendering:
- FFT spectrum: `(freqs: np.ndarray, magnitudes: np.ndarray)`
- Spectrogram/waterfall: `(times: np.ndarray, freqs: np.ndarray, power: np.ndarray)` (2D power array)
- Constellation: `(i_values: np.ndarray, q_values: np.ndarray)`

## Notes
- All arrays are NumPy, no pandas, to keep dependencies light.
- Every module function should accept the dict from the previous stage and return
  a dict that **includes all prior keys plus its own new ones** — this lets the
  GUI/integration layer inspect the full pipeline state at any point.
- If a module fails (e.g. can't classify modulation confidently), return the dict
  with an added `"error": "<message>"` key rather than raising, so the GUI can
  display it gracefully.
