"""
SIH 2026 - Signal Intelligence Toolkit
Upgraded Synthetic Signal Generator
Includes:
 - Modulation (BPSK, QPSK, 8PSK, 16QAM)
 - FEC (Hamming Code 7,4)
 - Bit Interleaving
 - SNR-Controlled AWGN Noise Injection
 - Ground-Truth Manifest Generation (JSON & CSV)
"""

import os
import json
import csv
import numpy as np
from pathlib import Path


# ============================================================
# 1. FEC & INTERLEAVING UTILS
# ============================================================

def hamming_7_4_encode_bits(bit_str: str) -> str:
    """Encodes 4-bit data chunks into 7-bit Hamming(7,4) codewords."""
    G = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=int)
    
    # Pad bit_str to multiple of 4
    pad_len = (4 - len(bit_str) % 4) % 4
    bit_str += "0" * pad_len
    
    encoded_bits = []
    for i in range(0, len(bit_str), 4):
        data = np.array([int(b) for b in bit_str[i:i+4]], dtype=int)
        codeword = np.dot(data, G) % 2
        encoded_bits.extend(codeword.tolist())
        
    return "".join(map(str, encoded_bits))


def interleave_bits(bit_str: str, cols: int = 8) -> str:
    """Block Interleaver: Arranges bits in matrix and transposes them."""
    pad_len = (cols - len(bit_str) % cols) % cols
    bit_str += "0" * pad_len
    
    rows = len(bit_str) // cols
    matrix = np.array([int(b) for b in bit_str]).reshape((rows, cols))
    interleaved = matrix.T.flatten()
    return "".join(map(str, interleaved))


# ============================================================
# 2. MODULATION SYMBOL MAPPING
# ============================================================

def text_to_bits(text: str) -> str:
    return "".join(f"{ord(c):08b}" for c in text)


def modulate_bits(bit_str: str, mod_type: str = "BPSK") -> np.ndarray:
    bits = np.array([int(b) for b in bit_str])
    
    if mod_type == "BPSK":
        # 1 bit per symbol
        symbols = 2 * bits - 1 + 0j
        
    elif mod_type == "QPSK":
        # 2 bits per symbol
        pad = (2 - len(bits) % 2) % 2
        bits = np.pad(bits, (0, pad))
        i_bits = 2 * bits[0::2] - 1
        q_bits = 2 * bits[1::2] - 1
        symbols = (i_bits + 1j * q_bits) / np.sqrt(2)
        
    elif mod_type == "8PSK":
        # 3 bits per symbol
        pad = (3 - len(bits) % 3) % 3
        bits = np.pad(bits, (0, pad))
        symbols = []
        for i in range(0, len(bits), 3):
            val = bits[i] * 4 + bits[i+1] * 2 + bits[i+2]
            phase = 2 * np.pi * val / 8
            symbols.append(np.exp(1j * phase))
        symbols = np.array(symbols)
        
    elif mod_type == "16QAM":
        # 4 bits per symbol
        pad = (4 - len(bits) % 4) % 4
        bits = np.pad(bits, (0, pad))
        mapping = {
            (0,0): -3, (0,1): -1, (1,1): 1, (1,0): 3
        }
        symbols = []
        for i in range(0, len(bits), 4):
            i_val = mapping[(bits[i], bits[i+1])]
            q_val = mapping[(bits[i+2], bits[i+3])]
            symbols.append((i_val + 1j * q_val) / np.sqrt(10))
        symbols = np.array(symbols)
    else:
        raise ValueError(f"Unsupported Modulation: {mod_type}")
        
    return symbols


# ============================================================
# 3. AWGN NOISE INJECTION
# ============================================================

def add_awgn_noise(signal: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Injects Complex Additive White Gaussian Noise based on SNR (dB)."""
    sig_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (target_snr_db / 10.0)
    noise_power = sig_power / snr_linear
    
    # Complex noise
    noise_i = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
    noise_q = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
    
    return signal + (noise_i + 1j * noise_q)


# ============================================================
# 4. MAIN SYNTHETIC GENERATOR
# ============================================================

def generate_synthetic_dataset(
    output_dir: str = "synthetic_dataset",
    num_samples_per_config: int = 1,
    sample_rate: float = 2_000_000.0
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    sync_word = "10101011"
    raw_payload_text = "SIH2026"
    payload_bits = text_to_bits(raw_payload_text)
    
    modulations = ["BPSK", "QPSK", "8PSK", "16QAM"]
    snr_levels = [0, 5, 10, 15, 20, 30]  # dB range
    
    manifest_records = []
    
    print("🚀 Starting Upgraded Synthetic Signal Generation...")
    
    file_id = 1
    for mod in modulations:
        for snr in snr_levels:
            for use_fec in [False, True]:
                for use_interleave in [False, True]:
                    
                    # Prepare bitstream
                    data_stream = payload_bits
                    if use_fec:
                        data_stream = hamming_7_4_encode_bits(data_stream)
                    if use_interleave:
                        data_stream = interleave_bits(data_stream)
                        
                    full_bitstream = sync_word + data_stream
                    
                    # Modulate & add noise
                    clean_symbols = modulate_bits(full_bitstream, mod_type=mod)
                    
                    # Oversample / repeat symbols for realistic IQ length
                    oversample_factor = 100
                    iq_signal = np.repeat(clean_symbols, oversample_factor)
                    
                    # Inject Noise
                    noisy_iq = add_awgn_noise(iq_signal, target_snr_db=snr).astype(np.complex64)
                    
                    # Save IQ File
                    file_name = f"sig_{file_id:03d}_{mod}_snr{snr}dB_fec{int(use_fec)}_itr{int(use_interleave)}.iq"
                    file_filepath = out_path / file_name
                    noisy_iq.tofile(file_filepath)
                    
                    # Record Manifest
                    manifest_records.append({
                        "id": file_id,
                        "file_name": file_name,
                        "file_path": str(file_filepath),
                        "modulation": mod,
                        "snr_db": snr,
                        "has_fec": use_fec,
                        "has_interleaving": use_interleave,
                        "sample_rate": sample_rate,
                        "num_samples": len(noisy_iq),
                        "raw_payload_text": raw_payload_text,
                        "sync_word": sync_word
                    })
                    
                    file_id += 1

    # Save JSON Manifest
    manifest_json_path = out_path / "manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=4)
        
    # Save CSV Manifest
    manifest_csv_path = out_path / "manifest.csv"
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_records[0].keys())
        writer.writeheader()
        writer.writerows(manifest_records)

    print(f"✅ Generation Complete!")
    print(f" - Total Signals Generated: {len(manifest_records)}")
    print(f" - Files Saved in: {out_path.resolve()}")
    print(f" - Ground-Truth Manifest: {manifest_json_path.resolve()}")


if __name__ == "__main__":
    generate_synthetic_dataset()