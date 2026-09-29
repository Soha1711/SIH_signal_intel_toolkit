# AstraWave — Automated Signal Intelligence & Demodulation Toolkit

A GUI-based signal intelligence toolkit that processes raw `.IQ` and `.WAV` recordings and provides an end-to-end workflow for signal analysis, modulation classification, demodulation, error recovery, and payload extraction.

## Overview

AstraWave converts raw I/Q recordings into meaningful signal information through an integrated DSP and ML pipeline.

The toolkit supports:

- `.IQ` and `.WAV` signal ingestion
- Common complex I/Q signal representation
- Signal preprocessing and normalization
- Sampling and signal parameter analysis
- FFT, PSD, waterfall, and constellation visualization
- ML-based modulation classification
- Confidence-based classification review
- Manual modulation override
- Symbol demodulation
- Block de-interleaving
- Hamming (7,4) FEC decoding
- Bitstream synchronization and correlation
- Frame detection and payload extraction
- ASCII/text decoding
- GUI-based visualization and results

## System Pipeline

```text
.IQ / .WAV Input
       ↓
File Ingestion & Preprocessing
       ↓
Signal Analysis & Parameter Estimation
       ↓
ML Modulation Classification
       ↓
Confidence Gate
   ↙         ↘
Auto Accept   Manual Review / Override
       ↓
Demodulation
       ↓
De-interleaving
       ↓
FEC Decoding
       ↓
Bitstream Correlation & Framing
       ↓
Payload Extraction
       ↓
ASCII / Text Output