# Module audit (auto-generated)
Fill the last column by hand: does it follow docs/interface_contract.md? What is missing?

## ingestion/

### ingestion/demod_payload.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `class PayloadProcessor: demodulate_bits, find_sync_word, decode_ascii_payload, extract_full_frame, extract_payload` | - |  |

### ingestion/iq_wav_reader.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `def read_wav(filepath)` | duration_sec, sample_rate, samples, source_format |  |
| `def read_iq(filepath, sample_rate, dtype)` | duration_sec, sample_rate, samples, source_format |  |
| `def remove_dc(signal)` | - |  |
| `def normalize_signal(signal, method)` | - |  |
| `def preprocess_signal(signal, normalize)` | - |  |
| `def load_signal_file(filepath, sample_rate, iq_dtype, preprocess)` | duration_sec, sample_rate, samples |  |

### ingestion/iq_wav_reader_backup.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `def read_wav(filepath)` | duration_sec, sample_rate, samples, source_format |  |
| `def read_iq(filepath, sample_rate, dtype)` | duration_sec, sample_rate, samples, source_format |  |
| `def remove_dc(signal)` | - |  |
| `def normalize_signal(signal, method)` | - |  |
| `def preprocess_signal(signal, normalize)` | - |  |
| `def load_signal_file(filepath, sample_rate, iq_dtype, preprocess)` | duration_sec, sample_rate, samples |  |

## param_estimation/

### param_estimation/demod_payload.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `class PayloadProcessor: center_frequency_shift, extract_bits_and_payload` | - |  |

### param_estimation/export_manager.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `class SignalExporter: export_json, export_csv, export_payload_binary, export_all` | - |  |

### param_estimation/modulation_classifier.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `def generate_bpsk()` | - |  |
| `def generate_qpsk()` | - |  |
| `def generate_8psk()` | - |  |
| `def generate_16qam()` | - |  |
| `def generate_64qam()` | - |  |
| `def get_ideal_constellation(mod_type)` | - |  |
| `def detect_modulation_type(i_data, q_data)` | - |  |
| `class ModulationClassifier: classify` | - |  |

### param_estimation/parameter_extractor.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `class ParameterExtractor: compute_snr, compute_bandwidth, get_full_extraction_report` | modulation_type, sample_rate |  |

### param_estimation/signal_analysis.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `def predict_modulation(iq_signal)` | - |  |
| `def compute_waveform(signal, num_points)` | - |  |
| `def compute_fft(signal, sample_rate)` | - |  |
| `def compute_psd(signal, sample_rate)` | - |  |
| `def find_peak_frequency(signal, sample_rate)` | - |  |
| `def estimate_bandwidth(signal, sample_rate)` | - |  |
| `def estimate_snr(signal)` | - |  |
| `def analyze_signal(input_data, sample_rate)` | confidence, modulation_type, sample_rate, samples |  |

### param_estimation/symbol_rate_estimator.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `class SymbolRateEstimator: estimate_symbol_rate` | - |  |
| `def estimate_symbol_rate(signal, sample_rate)` | - |  |

### param_estimation/test_signal_analysis.py
_no public functions/classes_

### param_estimation/test_waterfall_constellation.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `def test_waterfall_output()` | - |  |
| `def test_constellation_output()` | - |  |
| `def test_complete_analysis()` | - |  |

### param_estimation/train_classifier.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `def extract_signal_features(iq_signal)` | - |  |
| `def train_modulation_classifier(dataset_dir)` | - |  |

### param_estimation/waterfall_constellation.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `def compute_waterfall(signal, sample_rate, nperseg, noverlap)` | - |  |
| `def compute_constellation(signal, max_points)` | - |  |
| `def analyze_waterfall_constellation(signal, sample_rate, nperseg, noverlap, max_constellation_points)` | - |  |
| `def compute_waterfall(signal, sample_rate, nperseg, noverlap)` | - |  |
| `def compute_constellation(signal, max_points)` | - |  |
| `def analyze_waterfall_constellation(signal, sample_rate, nperseg, noverlap, max_constellation_points)` | - |  |

## demodulation/

### demodulation/robust_demodulator.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `def extract_symbols(iq_samples, samples_per_symbol)` | - |  |
| `def demodulate_bpsk(symbols)` | - |  |
| `def demodulate_qpsk(symbols)` | - |  |
| `def demodulate_8psk(symbols)` | - |  |
| `def demodulate_16qam(symbols)` | - |  |
| `def demodulate_symbols(iq_samples, modulation_type, samples_per_symbol)` | - |  |
| `def test_bpsk()` | - |  |
| `def test_qpsk()` | - |  |
| `def test_8psk()` | - |  |
| `def test_16qam()` | - |  |

### demodulation/robust_pipeline.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `def load_iq_file(file_path)` | - |  |
| `def classify_and_demodulate(iq_samples, classifier)` | confidence, modulation_type |  |
| `def remove_sync_word(bit_string)` | - |  |
| `def process_iq_file(file_path, classifier)` | confidence, modulation_type |  |
| `def decode_fec_message(demodulated_bits, has_fec, has_interleaving, interleaver_cols, fec_scheme)` | decoded_bits |  |
| `def test_complete_fec_pipeline()` | confidence, decoded_bits, modulation_type |  |

## deinterleave_fec/

### deinterleave_fec/bitstream_correlation.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `class CorrelationResult: sync_found, get` | - |  |
| `class BitstreamCorrelator: to_bipolar, correlate, correlate_multi_candidate` | - |  |
| `class FrameParser: parse_standard_header, split_frame` | header_end, header_start, payload_end, payload_start, sync_word_matched |  |
| `def correlate_and_frame(stage_data, sync_pattern, header_length_bits, confidence_threshold, structured_header, auto_invert)` | - |  |

### deinterleave_fec/fec_decoder.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `def hamming_7_4_encode_bits(bit_str)` | - |  |
| `def hamming_7_4_decode_bits(bit_str, original_length)` | - |  |
| `def parity(value)` | - |  |
| `def encoder_output(state, input_bit)` | - |  |
| `def next_state(state, input_bit)` | - |  |
| `def hamming_distance(received, expected)` | - |  |
| `def viterbi_decode(received_bits, terminate)` | - |  |
| `def bits_to_ascii(bit_string, allow_truncated)` | - |  |
| `class FECDecoder: decode` | - |  |
| `def process_fec_decoding(stage_data, fec_scheme, original_length)` | decoded_bits, error_count, fec_scheme |  |
| `def test_deinterleaver()` | - |  |
| `def test_hamming_fec()` | - |  |
| `def test_viterbi_fec()` | - |  |

### deinterleave_fec/interleaver.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `class BlockInterleaver: interleave, deinterleave` | - |  |
| `class ConvolutionalInterleaver: interleave, deinterleave` | - |  |
| `class DiagonalInterleaver: interleave, deinterleave` | - |  |
| `class PseudoRandomInterleaver: interleave, deinterleave` | - |  |
| `def interleave(bits, cols)` | - |  |
| `def deinterleave(bits, cols, original_length)` | - |  |
| `def interleave_bits(bit_str, cols)` | - |  |
| `def deinterleave_bits(bit_str, cols, original_length)` | - |  |
| `def process_deinterleaving(stage_data, interleaver_type, cols, original_length)` | deinterleaved_bits, interleaver_type |  |

## gui/

### gui/ingestion_test_gui.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `class SmoothScrollArea: setWidget, eventFilter, wheelEvent, keyPressEvent` | - |  |
| `class SignalIntelligenceGUI: apply_dark_theme, setup_ui, browse_file, load_signal, display_signal_information, display_parameters, format_frequency, calculate_bandwidth, calculate_snr, export_json, export_csv, export_plot` | confidence, sample_rate, samples |  |
| `def main()` | - |  |

### gui/signal_analysis_widget.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `class SignalAnalysisWidget: setup_ui, process_and_update_gui, display_analysis, plot_waveform, plot_fft, plot_psd, plot_waterfall, plot_constellation, export_csv, export_json, save_plot, clear_plots` | sample_rate, samples |  |

### gui/signal_analysis_widget_backup.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `class SignalAnalysisWidget: setup_ui, display_analysis, plot_row1, plot_waterfall, plot_constellation, clear_plots` | sample_rate, samples |  |

### gui/signal_analysis_widget_backup2.py
| Signature | Contract keys mentioned | Follows contract? / notes |
|---|---|---|
| `class SignalAnalysisWidget: setup_ui, display_analysis, plot_waveform, plot_fft, plot_psd, plot_waterfall, plot_constellation, clear_plots` | sample_rate, samples |  |
