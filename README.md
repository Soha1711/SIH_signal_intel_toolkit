# SIH — Automated Signal Intelligence & Demodulation Toolkit

GUI-based tool that takes a raw `.IQ` or `.WAV` recording and automatically:
detects signal parameters → demodulates → de-interleaves → FEC-decodes →
correlates the bitstream to find header/payload structure.

**Team:** Soha (Parameter Extraction + Integration Lead) · Honey (Demodulation) ·
Tirth (De-interleaving + FEC) · Hrutu & Rahul (GUI + Correlation)

**Project window:** 25 Aug – 20 Sep 2026 · **Internal Hackathon:** 31 Aug 2026

## Pipeline

```
File Ingestion → Parameter Estimation → Demodulation → De-interleaving
→ FEC Decoding → Bitstream Correlation → GUI Output
```

## Repo structure

| Folder | Owner | Purpose |
|---|---|---|
| `ingestion/` | Soha | Reads `.IQ` / `.WAV`, normalises to common complex-sample array |
| `param_estimation/` | Soha | Sampling rate, symbol rate, modulation classification, FFT/waterfall |
| `demodulation/` | Honey | FSK / PSK / QAM demodulators, constellation plots |
| `deinterleave_fec/` | Tirth | Block/Convolutional/Diagonal/PN de-interleaving, Viterbi/RS/LDPC decoding |
| `gui/` | Hrutu & Rahul | PyQt5 GUI shell, plot widgets, bitstream correlation, export |
| `docs/` | All | Interface contracts, architecture notes, PPT source |
| `sample_data/` | All | Test `.IQ` / `.WAV` files with known parameters |
| `tests/` | All | Unit tests per module |

## Getting started

See [SETUP.md](SETUP.md) for environment setup.
See [docs/interface_contract.md](docs/interface_contract.md) for how data passes between modules — read this before writing any module code.

## Status tracking

Progress is tracked on the repo's GitHub Projects board (see Issues/Projects tab).
