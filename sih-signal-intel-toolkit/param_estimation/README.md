# Parameter Estimation — Owner: Soha

Takes the output of `ingestion/iq_wav_reader.py` and estimates:
- Sampling frequency confirmation / symbol rate
- Occupied bandwidth
- Modulation type classification (FSK/PSK/QAM/ASK)
- FFT spectrum + spectrogram/waterfall data for plotting

See `docs/interface_contract.md` (Step 2) for exact input/output format.
