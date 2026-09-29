"""
orchestrator.py  --  TEMPLATE for Person D (put it at repo root or in a new pipeline/ folder)

Follows docs/interface_contract.md:
  * every stage takes the state dict and returns a dict that includes all prior keys + new ones
  * a failing stage sets "error" instead of crashing the GUI

HOW TO USE
  1. Audit each module folder and find the real function that does each stage.
  2. Replace the body of each `stage_*` adapter marked TODO with a real import + call.
  3. Anything still raising NotImplementedError shows up in the GUI as "not wired yet"
     instead of crashing, so you can integrate one stage at a time.

Stages written for you (no teammate code needed): input validation, confidence gate,
payload extraction, bits->ASCII with a printable-text check.
"""
import os
import time
import traceback

import numpy as np
from ingestion.iq_wav_reader import load_signal_file
from param_estimation.modulation_classifier import ModulationClassifier
from demodulation.robust_demodulator import demodulate_symbols
from deinterleave_fec.interleaver import deinterleave_bits
from deinterleave_fec.fec_decoder import process_fec_decoding
from deinterleave_fec.bitstream_correlation import correlate_and_frame

CONFIDENCE_THRESHOLD = 0.60          # below this -> flagged for manual review
SUPPORTED_EXTENSIONS = {".iq", ".wav"}
PRINTABLE_OK_RATIO = 0.90            # >= 90% printable characters => "looks like real text"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def validate_input_file(path):
    """Return an error string, or None if the file looks usable."""
    if not path or not os.path.exists(path):
        return f"File not found: {path}"
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    if os.path.getsize(path) == 0:
        return "File is empty."
    return None


def bits_to_ascii(bits, msb_first=True):
    """Convert a 0/1 array to text. Returns (text, printable_ratio)."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    usable = (len(bits) // 8) * 8
    if usable == 0:
        return "", 0.0
    packed = np.packbits(bits[:usable].reshape(-1, 8), axis=1, bitorder="big" if msb_first else "little")
    raw = packed.ravel()
    printable = np.sum(((raw >= 32) & (raw <= 126)) | np.isin(raw, [9, 10, 13]))
    text = bytes(raw.tolist()).decode("ascii", errors="replace")
    return text, float(printable) / len(raw)


def _run_stage(name, fn, state):
    """Run one stage safely. Never raises. Records timing + status in state['stage_log']."""
    t0 = time.perf_counter()
    ok, err = True, None
    try:
        out = fn(dict(state))                       # copy: a bad stage can't corrupt earlier keys
        if not isinstance(out, dict):
            raise TypeError(f"stage returned {type(out).__name__}, expected dict")
        if out.get("error"):
            ok, err = False, out["error"]
        state.update(out)
    except NotImplementedError as exc:
        ok, err = False, f"not wired yet: {exc}"
    except Exception as exc:                        # noqa: BLE001 - GUI must never crash
        ok, err = False, f"{type(exc).__name__}: {exc}"
        state.setdefault("debug_traces", {})[name] = traceback.format_exc()
    state.setdefault("stage_log", []).append(
        {"stage": name, "ok": ok, "error": err, "seconds": round(time.perf_counter() - t0, 4)}
    )
    if not ok:
        state["error"] = f"[{name}] {err}"
    return ok


# ----------------------------------------------------------------------------
# Stage adapters  -- TODO: wire these to your teammates' real functions
# ----------------------------------------------------------------------------
def _read_signal_fallback(path, sample_rate):
    """Working reader so the pipeline runs before ingestion/ is wired.
    .iq  -> int16 interleaved I,Q (same format run_pipeline.py reads); sample rate must be supplied
    .wav -> 2 channels = I,Q ; 1 channel = real signal (imag = 0); sample rate from header"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".iq":
        if not sample_rate:
            return {"error": "Sample rate is required for .iq files (no header). Enter it in the GUI."}
        raw = np.fromfile(path, dtype=np.int16)
        raw = raw[: (len(raw) // 2) * 2]
        samples = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
        rate, fmt = float(sample_rate), "iq"
    else:
        import soundfile as sf                              # listed in SETUP.md
        data, rate = sf.read(path, always_2d=True)
        samples = data[:, 0] + 1j * (data[:, 1] if data.shape[1] > 1 else 0.0)
        fmt = "wav"
    if len(samples) < 64:
        return {"error": f"File too short to analyse ({len(samples)} samples)."}
    return {"samples": samples.astype(np.complex64), "sample_rate": float(rate),
            "source_format": fmt, "duration_sec": len(samples) / float(rate)}


def stage_ingest(state):
    file_path = state["file_path"]
    sample_rate = state.get("sample_rate")

    result = load_signal_file(
        filepath=file_path,
        sample_rate=sample_rate,
        iq_dtype="float32",
        preprocess=True,
    )

    state.update(result)

    if "samples" in result:
        state["samples"] = result["samples"]

    if "sample_rate" in result:
        state["sample_rate"] = result["sample_rate"]

    return state


def stage_param_estimation(state):
    samples = state["samples"]

    classifier = ModulationClassifier()

    modulation_type, confidence = classifier.classify(samples)

    state["modulation_type"] = str(modulation_type)
    state["confidence"] = float(confidence)

    return state


def stage_demodulate(state):
    samples = state["samples"]
    modulation_type = state["modulation_type"]

    if modulation_type in ("Unknown", "None", ""):
        raise ValueError(
            f"Invalid modulation type: {modulation_type}"
        )

    demodulated_bits = demodulate_symbols(
        samples,
        modulation_type,
        samples_per_symbol=100,
    )

    state["bitstream"] = demodulated_bits
    state["demodulated_bits"] = demodulated_bits

    return state


def stage_deinterleave(state):
    """
    De-interleave only the payload portion.

    Dataset format:
        sync_word + interleaved_payload

    The sync word is NOT interleaved.
    """

    bitstream = state["bitstream"]

    # Normalize bitstream to a clean 0/1 string
    if isinstance(bitstream, str):
        bits = "".join(
            b for b in bitstream
            if b in "01"
        )
    else:
        bits = "".join(
            map(
                str,
                np.asarray(bitstream)
                .astype(np.uint8)
                .ravel()
            )
        )

    # Dataset sync word
    sync_word = state.get("sync_word", "10101011")
    sync_len = len(sync_word)

    # ---------------------------------------------------------
    # Case 1: No interleaving
    # ---------------------------------------------------------
    if not state.get("has_interleaving", False):

        arr = np.fromiter(
            (int(b) for b in bits),
            dtype=np.uint8
        )

        state["deinterleaved_bits"] = arr.copy()
        state["deinterleaved_bitstring"] = bits
        state["interleaver_type"] = "none"
        state["sync_word"] = sync_word

        return state

    # ---------------------------------------------------------
    # Case 2: Interleaving enabled
    # ---------------------------------------------------------

    # Sync word is NOT interleaved
    sync = bits[:sync_len]

    # Only payload was interleaved
    payload = bits[sync_len:]

    # De-interleave payload
    deinterleaved_payload = deinterleave_bits(
        payload,
        cols=8,
        original_length=len(payload)
    )

    # Reconstruct:
    # sync + recovered payload
    reconstructed = sync + deinterleaved_payload

    # Convert reconstructed bit string to numpy array
    arr = np.fromiter(
        (int(b) for b in reconstructed),
        dtype=np.uint8
    )

    state["sync_word"] = sync
    state["deinterleaved_bits"] = arr
    state["deinterleaved_bitstring"] = reconstructed
    state["interleaver_type"] = "block"

    return state


def stage_fec_decode(state):
    """
    Decode FEC on the payload only.

    Dataset format:
        sync_word + FEC_encoded_payload

    For SIH2026:
        payload = 56 bits
        Hamming(7,4) -> 98 coded bits
        sync = 8 bits
        total = 106 bits
    """
    from deinterleave_fec.fec_decoder import process_fec_decoding

    # Normalize the deinterleaved stream to a bitstring
    bitstream = state.get("deinterleaved_bitstring")

    if bitstream is None:
        bits = state.get("deinterleaved_bits", state.get("bitstream", ""))
        if isinstance(bits, str):
            bitstream = "".join(b for b in bits if b in "01")
        else:
            bitstream = "".join(
                map(str, np.asarray(bits, dtype=np.uint8).ravel())
            )

    sync_word = state.get("sync_word", "10101011")
    sync_len = len(sync_word)

    # No FEC: nothing to decode.
    if not state.get("has_fec", False):
        arr = np.fromiter((int(b) for b in bitstream), dtype=np.uint8)

        state["decoded_bits"] = arr
        state["decoded_bitstring"] = bitstream
        state["fec_scheme"] = "none"
        state["error_count"] = 0
        state["ascii_message"] = ""
        return state

    # FEC applies ONLY to payload, not sync word.
    sync = bitstream[:sync_len]
    payload = bitstream[sync_len:]

    # Hamming(7,4):
    # 56 original payload bits -> 98 coded bits.
    #
    # Ignore any trailing demodulation artifacts beyond the expected
    # complete Hamming codewords.
    if state.get("fec_scheme", "hamming").lower() == "hamming":
        expected_coded_length = 98

        if len(payload) < expected_coded_length:
            raise ValueError(
                f"Not enough Hamming payload bits: "
                f"got {len(payload)}, expected {expected_coded_length}"
            )

        payload = payload[:expected_coded_length]

    # Decode payload only.
    fec_input = {
        "deinterleaved_bits": np.fromiter(
            (int(b) for b in payload),
            dtype=np.uint8
        )
    }

    result = process_fec_decoding(
        fec_input,
        fec_scheme=state.get("fec_scheme", "hamming"),
        original_length=56,
    )

    decoded_payload = result.get("decoded_bitstring", "")

    # Reconstruct:
    # sync + decoded payload
    reconstructed = sync + decoded_payload

    state["decoded_bits"] = np.fromiter(
        (int(b) for b in reconstructed),
        dtype=np.uint8
    )
    state["decoded_bitstring"] = reconstructed
    state["fec_scheme"] = state.get("fec_scheme", "hamming")
    state["error_count"] = result.get("error_count", 0)
    state["ascii_message"] = result.get("ascii_message", "")

    return state

    result = process_fec_decoding(
        state,
        fec_scheme=state.get("fec_scheme", "hamming"),
    )

    state.update(result)
    return state


def stage_correlate(state):
    result = correlate_and_frame(
        state,
        sync_pattern=state.get("sync_pattern", "10101011"),
        header_length_bits=state.get("header_length_bits", 0),
        confidence_threshold=0.75,
        structured_header=False,
        auto_invert=True,
    )

    state.update(result)

    return state


# ----------------------------------------------------------------------------
# Stages owned by Person D (fully implemented)
# ----------------------------------------------------------------------------
def stage_confidence_gate(state):
    """Flag low classifier confidence. Flag-and-continue: pipeline keeps going, GUI shows a banner."""
    conf = state.get("confidence")
    state.setdefault("warnings", [])
    if state.get("modulation_source") == "manual":
        return state
    if conf is None:
        state["warnings"].append("Classifier returned no confidence score.")
        state["needs_review"] = True
    elif conf < CONFIDENCE_THRESHOLD:
        state["warnings"].append(
            f"Low classifier confidence ({conf:.0%}) for '{state.get('modulation_type')}'. "
            "Result may be unreliable - consider choosing the modulation manually."
        )
        state["needs_review"] = True
    else:
        state["needs_review"] = False
    return state


def stage_payload_to_text(state):
    """Cut payload using correlation indices, convert to ASCII, and sanity-check it."""
    bits = np.asarray(state["decoded_bits"], dtype=np.uint8)
    ps, pe = state.get("payload_start"), state.get("payload_end")
    if ps is not None and pe is not None and 0 <= ps < pe <= len(bits):
        payload = bits[ps:pe]
    else:
        payload = bits
        state.setdefault("warnings", []).append("No valid payload span found - decoding all bits as text.")
    text, ratio = bits_to_ascii(payload)
    state.update(
        payload_bits=payload,
        ascii_text=text,
        printable_ratio=ratio,
        text_looks_valid=ratio >= PRINTABLE_OK_RATIO,
    )
    # Test/demo only: if the caller supplied ground truth, compare it.
    expected = state.get("expected_text")
    if expected is not None:
        state["matches_expected"] = text.strip("\x00").startswith(expected)
    return state


PIPELINE = [
    ("ingest", stage_ingest),
    ("param_estimation", stage_param_estimation),
    ("confidence_gate", stage_confidence_gate),
    ("demodulate", stage_demodulate),
    ("deinterleave", stage_deinterleave),
    ("fec_decode", stage_fec_decode),
    ("correlate", stage_correlate),
    ("payload_to_text", stage_payload_to_text),
]


# ----------------------------------------------------------------------------
# Public entry point -- the GUI calls ONLY this
# ----------------------------------------------------------------------------
def run_full_pipeline(file_path,
    sample_rate=None,
    modulation_override=None,
    expected_text=None,
    has_fec=None,
    has_interleaving=None,
    fec_scheme="hamming",
    interleaver_type="block",
):
    """
    Returns the full state dict. Always returns - never raises.
    """
    state = {
        "file_path": file_path,
        "stage_log": [],
        "warnings": [],
    }

    # Explicit pipeline configuration
    if has_fec is not None:
        state["has_fec"] = bool(has_fec)

    if has_interleaving is not None:
        state["has_interleaving"] = bool(has_interleaving)

    state["fec_scheme"] = fec_scheme
    state["interleaver_type"] = interleaver_type

    if sample_rate:
        state["sample_rate"] = float(sample_rate)

    if expected_text is not None:
        state["expected_text"] = expected_text

    problem = validate_input_file(file_path)

    if problem:
        state["error"] = problem
        return state

    for name, fn in PIPELINE:
        if name == "confidence_gate" and modulation_override:
            state["modulation_type"] = modulation_override
            state["modulation_source"] = "manual"

        if not _run_stage(name, fn, state):
            break

    return state



if __name__ == "__main__":
    import sys
    result = run_full_pipeline(sys.argv[1] if len(sys.argv) > 1 else "sample_data/test_tone.iq")
    for entry in result["stage_log"]:
        print(entry)
    print("error:", result.get("error"))