"""
gui/pipeline_panel.py

Full-pipeline PyQt6 panel for the SIH Signal Intelligence Toolkit.

Features:
    - .IQ / .WAV signal selection
    - Automatic modulation classification
    - Manual modulation override
    - Automatic FEC/interleaving detection from synthetic-dataset filenames
    - Manual FEC/interleaving configuration
    - Background pipeline execution using QThread
    - Stage-by-stage status
    - Demodulated / de-interleaved / FEC-corrected bit views
    - Correlation / framing visualization
    - Decoded ASCII text display
"""

import html
import os
import re

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pipeline.orchestrator import run_full_pipeline


MAX_BITS_SHOWN = 4000

# Current trained/evaluated classifier supports these modulations.
MODULATIONS = [
    "Auto (classifier)",
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
]

FEC_OPTIONS = [
    "Auto (filename)",
    "None",
    "Hamming",
]

INTERLEAVING_OPTIONS = [
    "Auto (filename)",
    "None",
    "Block",
]


def detect_dataset_config(filepath):
    """
    Detect FEC/interleaving configuration from synthetic dataset filenames.

    Expected filename markers:
        _fec0_  -> FEC disabled
        _fec1_  -> Hamming FEC enabled
        _itr0_  -> interleaving disabled
        _itr1_  -> block interleaving enabled

    Returns:
        {
            "has_fec": bool,
            "has_interleaving": bool,
            "fec_scheme": "hamming" or "none",
            "interleaver_type": "block" or "none",
            "fec_detected": bool,
            "interleaving_detected": bool,
        }

    If a marker is absent, the corresponding feature defaults to disabled.
    The GUI still allows explicit manual configuration.
    """
    name = os.path.basename(filepath or "").lower()

    fec_match = re.search(r"_fec([01])(?:_|\.|$)", name)
    itr_match = re.search(r"_itr([01])(?:_|\.|$)", name)

    fec_detected = fec_match is not None
    interleaving_detected = itr_match is not None

    has_fec = bool(fec_match and fec_match.group(1) == "1")
    has_interleaving = bool(
        itr_match and itr_match.group(1) == "1"
    )

    return {
        "has_fec": has_fec,
        "has_interleaving": has_interleaving,
        "fec_scheme": "hamming" if has_fec else "none",
        "interleaver_type": "block" if has_interleaving else "none",
        "fec_detected": fec_detected,
        "interleaving_detected": interleaving_detected,
    }


def resolve_pipeline_config(
    filepath,
    fec_selection,
    interleaving_selection,
):
    """
    Resolve GUI selections into the exact arguments expected by
    run_full_pipeline().

    Auto mode:
        Uses _fec0/_fec1 and _itr0/_itr1 filename markers.

    Manual mode:
        Uses the user's explicit selection.
    """
    detected = detect_dataset_config(filepath)

    # ---------------------------
    # FEC
    # ---------------------------
    if fec_selection == "Auto (filename)":
        has_fec = detected["has_fec"]
        fec_scheme = detected["fec_scheme"]
        fec_source = (
            "filename"
            if detected["fec_detected"]
            else "auto default (FEC off)"
        )
    elif fec_selection == "Hamming":
        has_fec = True
        fec_scheme = "hamming"
        fec_source = "manual"
    else:
        has_fec = False
        fec_scheme = "none"
        fec_source = "manual"

    # ---------------------------
    # Interleaving
    # ---------------------------
    if interleaving_selection == "Auto (filename)":
        has_interleaving = detected["has_interleaving"]
        interleaver_type = detected["interleaver_type"]
        interleaver_source = (
            "filename"
            if detected["interleaving_detected"]
            else "auto default (interleaving off)"
        )
    elif interleaving_selection == "Block":
        has_interleaving = True
        interleaver_type = "block"
        interleaver_source = "manual"
    else:
        has_interleaving = False
        interleaver_type = "none"
        interleaver_source = "manual"

    return {
        "has_fec": has_fec,
        "has_interleaving": has_interleaving,
        "fec_scheme": fec_scheme,
        "interleaver_type": interleaver_type,
        "fec_source": fec_source,
        "interleaver_source": interleaver_source,
        "detected": detected,
    }


class PipelineWorker(QThread):
    """Runs the complete pipeline outside the GUI thread."""

    finished_state = pyqtSignal(dict)

    def __init__(
        self,
        path,
        sample_rate,
        override,
        has_fec,
        has_interleaving,
        fec_scheme,
        interleaver_type,
        parent=None,
    ):
        super().__init__(parent)

        self.path = path
        self.sample_rate = sample_rate
        self.override = override
        self.has_fec = has_fec
        self.has_interleaving = has_interleaving
        self.fec_scheme = fec_scheme
        self.interleaver_type = interleaver_type

    def run(self):
        try:
            state = run_full_pipeline(
                self.path,
                sample_rate=self.sample_rate,
                modulation_override=self.override,
                has_fec=self.has_fec,
                has_interleaving=self.has_interleaving,
                fec_scheme=self.fec_scheme,
                interleaver_type=self.interleaver_type,
            )

            # Preserve the effective GUI configuration in the result so
            # the Summary tab and parent dashboard can display it.
            state["gui_has_fec"] = self.has_fec
            state["gui_has_interleaving"] = self.has_interleaving
            state["gui_fec_scheme"] = self.fec_scheme
            state["gui_interleaver_type"] = self.interleaver_type

        except Exception as exc:
            state = {
                "error": f"Unexpected failure: {exc}",
                "stage_log": [],
            }

        self.finished_state.emit(state)


def _bits_text(bits):
    """Convert pipeline bit data into a readable 0/1 string safely."""
    if bits is None:
        return ""

    if isinstance(bits, str):
        return "".join(ch for ch in bits if ch in "01")

    if isinstance(bits, (bytes, bytearray, memoryview)):
        return "".join(f"{b:08b}" for b in bytes(bits))

    if isinstance(bits, (int, np.integer)):
        value = int(bits)
        if value < 0:
            return ""
        return format(value, "b") if value else "0"

    try:
        arr = np.asarray(bits)
    except Exception:
        return str(bits)

    if arr.ndim == 0:
        try:
            value = int(arr.item())
            return format(value, "b") if value >= 0 else ""
        except Exception:
            return str(arr.item())

    if arr.dtype.kind in ("U", "S", "O"):
        values = arr.ravel().tolist()
        out = []

        for value in values:
            if isinstance(value, (int, np.integer)):
                out.append("1" if int(value) else "0")
            else:
                out.extend(ch for ch in str(value) if ch in "01")

        return "".join(out)

    values = arr.ravel().tolist()
    out = []

    for value in values:
        try:
            out.append("1" if int(value) != 0 else "0")
        except Exception:
            pass

    return "".join(out)


class PipelinePanel(QWidget):

    pipeline_finished = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.path = None
        self.worker = None

        self._build_ui()

    # ==================================================================
    # UI
    # ==================================================================

    def _build_ui(self):
        root = QVBoxLayout(self)

        # --------------------------------------------------------------
        # Top controls
        # --------------------------------------------------------------
        top = QHBoxLayout()

        self.btn_open = QPushButton("Open .IQ / .WAV ...")
        self.btn_open.clicked.connect(self.choose_file)

        self.lbl_file = QLabel("No file selected")

        self.spin_rate = QDoubleSpinBox()
        self.spin_rate.setRange(0, 1e10)
        self.spin_rate.setDecimals(0)
        self.spin_rate.setSuffix(" Hz")
        self.spin_rate.setSpecialValueText(
            "Sample rate: auto / from header"
        )

        self.combo_mod = QComboBox()
        self.combo_mod.addItems(MODULATIONS)
        self.combo_mod.setToolTip(
            "Auto uses the ML classifier. Select a modulation "
            "to manually override it."
        )

        self.btn_run = QPushButton("Run pipeline")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_pipeline)

        for widget in (
            self.btn_open,
            self.lbl_file,
            self.spin_rate,
            self.combo_mod,
            self.btn_run,
        ):
            top.addWidget(widget)

        top.setStretch(1, 1)
        root.addLayout(top)

        # --------------------------------------------------------------
        # Pipeline configuration
        # --------------------------------------------------------------
        config = QHBoxLayout()

        config.addWidget(QLabel("FEC:"))

        self.combo_fec = QComboBox()
        self.combo_fec.addItems(FEC_OPTIONS)
        self.combo_fec.setToolTip(
            "Auto reads _fec0/_fec1 from synthetic dataset filenames. "
            "For real files, choose the known FEC manually."
        )
        config.addWidget(self.combo_fec)

        config.addWidget(QLabel("Interleaving:"))

        self.combo_interleave = QComboBox()
        self.combo_interleave.addItems(INTERLEAVING_OPTIONS)
        self.combo_interleave.setToolTip(
            "Auto reads _itr0/_itr1 from synthetic dataset filenames. "
            "For real files, choose the known interleaving manually."
        )
        config.addWidget(self.combo_interleave)

        self.lbl_config = QLabel(
            "Configuration: not resolved"
        )
        self.lbl_config.setStyleSheet(
            "color:#777;"
        )

        config.addWidget(
            self.lbl_config,
            1,
        )

        root.addLayout(config)

        # --------------------------------------------------------------
        # Status banner
        # --------------------------------------------------------------
        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.hide()
        root.addWidget(self.banner)

        # --------------------------------------------------------------
        # Main splitter
        # --------------------------------------------------------------
        split = QSplitter(Qt.Orientation.Horizontal)

        self.stage_list = QListWidget()
        self.stage_list.setMaximumWidth(280)
        split.addWidget(self.stage_list)

        self.tabs = QTabWidget()
        self.txt = {}

        for name in (
            "Summary",
            "Demodulated bits",
            "De-interleaved bits",
            "FEC-corrected bits",
            "Correlation",
            "Decoded text",
        ):
            widget = (
                QTextEdit()
                if name == "Correlation"
                else QPlainTextEdit()
            )

            widget.setReadOnly(True)
            widget.setStyleSheet(
                "font-family: Consolas, 'Courier New', monospace;"
            )

            self.tabs.addTab(widget, name)
            self.txt[name] = widget

        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

    # ==================================================================
    # Banner
    # ==================================================================

    def _show_banner(self, text, kind):
        colours = {
            "error": "#8b1a1a;background:#fde8e8",
            "warn": "#7a5b00;background:#fff4d6",
            "ok": "#14532d;background:#e3f6e8",
        }

        self.banner.setStyleSheet(
            f"padding:8px;border-radius:4px;color:{colours[kind]};"
        )
        self.banner.setText(text)
        self.banner.show()

    # ==================================================================
    # Configuration display
    # ==================================================================

    def _refresh_config_preview(self):
        if not self.path:
            self.lbl_config.setText(
                "Configuration: select a file"
            )
            return

        config = resolve_pipeline_config(
            self.path,
            self.combo_fec.currentText(),
            self.combo_interleave.currentText(),
        )

        fec_text = (
            "Hamming"
            if config["has_fec"]
            else "None"
        )

        interleave_text = (
            "Block"
            if config["has_interleaving"]
            else "None"
        )

        self.lbl_config.setText(
            f"Configuration: FEC={fec_text} "
            f"({config['fec_source']}), "
            f"Interleaving={interleave_text} "
            f"({config['interleaver_source']})"
        )

    # ==================================================================
    # File selection
    # ==================================================================

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select signal file",
            "",
            "Signal files (*.iq *.IQ *.wav *.WAV)",
        )

        if path:
            self.path = path
            self.lbl_file.setText(
                os.path.basename(path)
            )
            self.btn_run.setEnabled(True)
            self._refresh_config_preview()

    # ==================================================================
    # Run pipeline
    # ==================================================================

    def run_pipeline(self):
        if not self.path:
            return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running...")
        self.stage_list.clear()
        self.banner.hide()

        rate = self.spin_rate.value() or None

        if self.combo_mod.currentIndex() == 0:
            override = None
        else:
            override = self.combo_mod.currentText()

        config = resolve_pipeline_config(
            self.path,
            self.combo_fec.currentText(),
            self.combo_interleave.currentText(),
        )

        self.lbl_config.setText(
            f"Running: FEC="
            f"{'Hamming' if config['has_fec'] else 'None'}, "
            f"Interleaving="
            f"{'Block' if config['has_interleaving'] else 'None'}"
        )

        self.worker = PipelineWorker(
            self.path,
            rate,
            override,
            config["has_fec"],
            config["has_interleaving"],
            config["fec_scheme"],
            config["interleaver_type"],
            self,
        )

        self.worker.finished_state.connect(
            self.on_finished
        )
        self.worker.start()

    # ==================================================================
    # Pipeline completed
    # ==================================================================

    def on_finished(self, state):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run pipeline")
        self.show_state(state)
        self.pipeline_finished.emit(state)

    # ==================================================================
    # Render state
    # ==================================================================

    def show_state(self, s):

        # --------------------------------------------------------------
        # Stage checklist
        # --------------------------------------------------------------
        self.stage_list.clear()

        for entry in s.get("stage_log", []):
            mark = "OK " if entry["ok"] else "FAIL"

            item = QListWidgetItem(
                f"[{mark}] {entry['stage']} "
                f"({entry['seconds']}s)"
            )

            if not entry["ok"]:
                item.setToolTip(
                    entry["error"] or ""
                )

            self.stage_list.addItem(item)

        # --------------------------------------------------------------
        # Banner
        # --------------------------------------------------------------
        if s.get("error"):
            self._show_banner(
                f"Pipeline stopped: {s['error']} "
                "(results from earlier stages are still shown)",
                "error",
            )

        elif s.get("needs_review"):
            self._show_banner(
                "; ".join(
                    s.get("warnings", [])
                )
                or "Flagged for manual review.",
                "warn",
            )

        elif s.get("text_looks_valid"):
            self._show_banner(
                "Decoded successfully - "
                "output looks like valid text.",
                "ok",
            )

        elif s.get("warnings"):
            self._show_banner(
                "; ".join(s["warnings"]),
                "warn",
            )

        # --------------------------------------------------------------
        # Classification status
        # --------------------------------------------------------------
        modulation = s.get(
            "modulation_type"
        )

        modulation_source = s.get(
            "modulation_source",
            "classifier",
        )

        confidence = s.get("confidence")

        if modulation_source == "manual":
            classification_source = "manual override"
            classification_status = "MANUAL OVERRIDE"

        elif confidence is None:
            classification_source = "classifier (AUTO)"
            classification_status = "NO CONFIDENCE SCORE"

        elif confidence < 0.60:
            classification_source = "classifier (AUTO)"
            classification_status = "REVIEW REQUIRED"

        else:
            classification_source = "classifier (AUTO)"
            classification_status = "ACCEPTED"

        confidence_text = (
            f"{confidence:.1%}"
            if isinstance(confidence, (int, float))
            else "-"
        )

        # --------------------------------------------------------------
        # Effective pipeline configuration
        # --------------------------------------------------------------
        has_fec = bool(
            s.get("gui_has_fec", False)
        )

        has_interleaving = bool(
            s.get("gui_has_interleaving", False)
        )

        fec_scheme = (
            s.get("gui_fec_scheme")
            or s.get("fec_scheme")
            or "none"
        )

        interleaver_type = (
            s.get("gui_interleaver_type")
            or s.get("interleaver_type")
            or "none"
        )

        # Update configuration label.
        self.lbl_config.setText(
            f"Effective configuration: "
            f"FEC={'Hamming' if has_fec else 'None'}, "
            f"Interleaving="
            f"{'Block' if has_interleaving else 'None'}"
        )

        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------
        fields = [
            (
                "File",
                os.path.basename(
                    s.get("file_path", "") or ""
                ),
            ),
            (
                "Format",
                s.get("source_format"),
            ),
            (
                "Sample rate (Hz)",
                s.get("sample_rate"),
            ),
            (
                "Duration (s)",
                s.get("duration_sec"),
            ),
            (
                "Modulation",
                modulation,
            ),
            (
                "Modulation source",
                classification_source,
            ),
            (
                "Classification status",
                classification_status,
            ),
            (
                "Classifier confidence",
                confidence_text,
            ),
            (
                "Symbol rate (Hz)",
                s.get("symbol_rate"),
            ),
            (
                "Occupied bandwidth (Hz)",
                s.get("occupied_bandwidth"),
            ),
            (
                "Interleaver",
                interleaver_type
                if has_interleaving
                else "none",
            ),
            (
                "FEC scheme",
                fec_scheme
                if has_fec
                else "none",
            ),
            (
                "Errors corrected",
                s.get("error_count"),
            ),
            (
                "Sync word matched",
                s.get("sync_word_matched"),
            ),
        ]

        self.txt["Summary"].setPlainText(
            "\n".join(
                f"{key:26s} "
                f"{value if value not in (None, '') else '-'}"
                for key, value in fields
            )
        )

        # --------------------------------------------------------------
        # Bit views
        # --------------------------------------------------------------
        self.txt["Demodulated bits"].setPlainText(
            _bits_text(
                s.get("bitstream")
            )
        )

        self.txt["De-interleaved bits"].setPlainText(
            _bits_text(
                s.get("deinterleaved_bits")
            )
        )

        self.txt["FEC-corrected bits"].setPlainText(
            _bits_text(
                s.get("decoded_bits")
            )
        )

        # --------------------------------------------------------------
        # Correlation
        # --------------------------------------------------------------
        self.txt["Correlation"].setHtml(
            self._correlation_html(s)
        )

        # --------------------------------------------------------------
        # Decoded text
        # --------------------------------------------------------------
        if "ascii_text" in s:

            printable_ratio = s.get(
                "printable_ratio",
                0,
            )

            looks_valid = s.get(
                "text_looks_valid",
                False,
            )

            body = (
                f"printable ratio: "
                f"{printable_ratio:.0%}   "
                f"looks like valid text: "
                f"{'yes' if looks_valid else 'NO'}"
                "\n\n"
                f"{s['ascii_text']}"
            )

        else:
            body = "(text stage did not run)"

        self.txt["Decoded text"].setPlainText(
            body
        )

    # ==================================================================
    # Correlation HTML
    # ==================================================================

    def _correlation_html(self, s):

        bits = s.get("decoded_bits")

        if bits is None:
            return (
                "<i>(not available - earlier stage "
                "did not finish)</i>"
            )

        bits_text = _bits_text(bits)
        bits_text = bits_text[:MAX_BITS_SHOWN]

        hs = s.get("header_start")
        he = s.get("header_end")
        ps = s.get("payload_start")
        pe = s.get("payload_end")

        if None in (hs, he, ps, pe):
            return (
                "<i>No header/payload boundaries found.</i>"
                "<br><br>"
                + html.escape(bits_text)
            )

        length = len(bits_text)

        hs = max(0, min(int(hs), length))
        he = max(hs, min(int(he), length))
        ps = max(he, min(int(ps), length))
        pe = max(ps, min(int(pe), length))

        def seg(start, end):
            return bits_text[start:end]

        return (
            "<p>"
            "Sync word matched: "
            f"<b>"
            f"{html.escape(str(s.get('sync_word_matched') or '-'))}"
            f"</b>"
            "</p>"

            "<p>"
            f"<span style='background:#ffe3b3'>"
            f"header [{hs}:{he}]"
            f"</span> "

            f"<span style='background:#c9f0d3'>"
            f"payload [{ps}:{pe}]"
            f"</span>"
            "</p>"

            "<p style='font-family:monospace'>"

            f"{html.escape(seg(0, hs))}"

            f"<span style='background:#ffe3b3'>"
            f"{html.escape(seg(hs, he))}"
            f"</span>"

            f"{html.escape(seg(he, ps))}"

            f"<span style='background:#c9f0d3'>"
            f"{html.escape(seg(ps, pe))}"
            f"</span>"

            f"{html.escape(seg(pe, len(bits_text)))}"

            "</p>"
        )


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    panel = PipelinePanel()
    panel.resize(1100, 700)
    panel.show()

    sys.exit(app.exec())
