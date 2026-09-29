"""
gui/pipeline_panel.py  --  STEP 5
A self-contained PyQt5 widget that runs the FULL pipeline and shows every intermediate stage.
Add it to your existing main window as a new tab:

    from gui.pipeline_panel import PipelinePanel
    tabs.addTab(PipelinePanel(), "Full Pipeline")          # or: PipelinePanel().show()

It only talks to pipeline.orchestrator.run_full_pipeline(), so it works with whatever
stages are wired so far and shows "not wired yet" for the rest.
"""
import html
import os

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt

from pipeline.orchestrator import run_full_pipeline

MAX_BITS_SHOWN = 4000          # keep the widgets fast on long recordings
MODULATIONS = ["Auto (classifier)", "FSK", "PSK", "QAM", "ASK"]


class PipelineWorker(QThread):
    """Runs the pipeline off the GUI thread so the window never freezes."""
    finished_state = pyqtSignal(dict)

    def __init__(self, path, sample_rate, override, parent=None):
        super().__init__(parent)
        self.path, self.sample_rate, self.override = path, sample_rate, override

    def run(self):
        try:
            state = run_full_pipeline(self.path, sample_rate=self.sample_rate, modulation_override=self.override)
        except Exception as exc:                       # last line of defence - should never trigger
            state = {"error": f"Unexpected failure: {exc}", "stage_log": []}
        self.finished_state.emit(state)


def _bits_text(bits, group=8, per_line=64):
    if bits is None:
        return "(not available - stage did not run)"
    bits = np.asarray(bits).astype(np.uint8).ravel()
    shown = "".join(map(str, bits[:MAX_BITS_SHOWN]))
    chunks = [shown[i:i + group] for i in range(0, len(shown), group)]
    lines = [" ".join(chunks[i:i + per_line // group]) for i in range(0, len(chunks), per_line // group)]
    suffix = f"\n... ({len(bits) - MAX_BITS_SHOWN} more bits)" if len(bits) > MAX_BITS_SHOWN else ""
    return f"{len(bits)} bits\n\n" + "\n".join(lines) + suffix


class PipelinePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.path, self.worker = None, None
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.btn_open = QPushButton("Open .IQ / .WAV ...")
        self.btn_open.clicked.connect(self.choose_file)
        self.lbl_file = QLabel("No file selected")
        self.spin_rate = QDoubleSpinBox()
        self.spin_rate.setRange(0, 1e10)
        self.spin_rate.setDecimals(0)
        self.spin_rate.setSuffix(" Hz")
        self.spin_rate.setSpecialValueText("Sample rate: auto / from header")
        self.combo_mod = QComboBox()
        self.combo_mod.addItems(MODULATIONS)
        self.combo_mod.setToolTip("Override the classifier if its confidence is low")
        self.btn_run = QPushButton("Run pipeline")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_pipeline)
        for w in (self.btn_open, self.lbl_file, self.spin_rate, self.combo_mod, self.btn_run):
            top.addWidget(w)
        top.setStretch(1, 1)
        root.addLayout(top)

        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.hide()
        root.addWidget(self.banner)

        split = QSplitter(Qt.Horizontal)
        self.stage_list = QListWidget()
        self.stage_list.setMaximumWidth(260)
        split.addWidget(self.stage_list)

        self.tabs = QTabWidget()
        self.txt = {}
        for name in ("Summary", "Demodulated bits", "De-interleaved bits", "FEC-corrected bits", "Correlation", "Decoded text"):
            widget = QTextEdit() if name == "Correlation" else QPlainTextEdit()
            widget.setReadOnly(True)
            widget.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")
            self.tabs.addTab(widget, name)
            self.txt[name] = widget
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

    def _show_banner(self, text, kind):
        colours = {"error": "#8b1a1a;background:#fde8e8", "warn": "#7a5b00;background:#fff4d6", "ok": "#14532d;background:#e3f6e8"}
        self.banner.setStyleSheet(f"padding:8px;border-radius:4px;color:{colours[kind]};")
        self.banner.setText(text)
        self.banner.show()

    # ------------------------------------------------------------------ actions
    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select signal file", "", "Signal files (*.iq *.IQ *.wav *.WAV)")
        if path:
            self.path = path
            self.lbl_file.setText(os.path.basename(path))
            self.btn_run.setEnabled(True)

    def run_pipeline(self):
        if not self.path:
            return
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running...")
        self.stage_list.clear()
        self.banner.hide()
        rate = self.spin_rate.value() or None
        override = None if self.combo_mod.currentIndex() == 0 else self.combo_mod.currentText()
        self.worker = PipelineWorker(self.path, rate, override, self)
        self.worker.finished_state.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, state):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run pipeline")
        self.show_state(state)

    # ------------------------------------------------------------------ rendering
    def show_state(self, s):
        # stage checklist
        self.stage_list.clear()
        for entry in s.get("stage_log", []):
            mark = "OK " if entry["ok"] else "FAIL"
            item = QListWidgetItem(f"[{mark}] {entry['stage']}  ({entry['seconds']}s)")
            if not entry["ok"]:
                item.setToolTip(entry["error"] or "")
            self.stage_list.addItem(item)

        # banner: error > low confidence > success
        if s.get("error"):
            self._show_banner(f"Pipeline stopped: {s['error']}  (results from earlier stages are still shown)", "error")
        elif s.get("needs_review"):
            self._show_banner("; ".join(s.get("warnings", [])) or "Flagged for manual review.", "warn")
        elif s.get("text_looks_valid"):
            self._show_banner("Decoded successfully - output looks like valid text.", "ok")
        elif s.get("warnings"):
            self._show_banner("; ".join(s["warnings"]), "warn")

        # summary
        fields = [
            ("File", os.path.basename(s.get("file_path", "") or "")), ("Format", s.get("source_format")),
            ("Sample rate (Hz)", s.get("sample_rate")), ("Duration (s)", s.get("duration_sec")),
            ("Modulation", s.get("modulation_type")), ("Modulation source", s.get("modulation_source", "classifier")),
            ("Classifier confidence", f"{s['confidence']:.1%}" if isinstance(s.get("confidence"), (int, float)) else None),
            ("Symbol rate (Hz)", s.get("symbol_rate")), ("Occupied bandwidth (Hz)", s.get("occupied_bandwidth")),
            ("Interleaver", s.get("interleaver_type")), ("FEC scheme", s.get("fec_scheme")),
            ("Errors corrected", s.get("error_count")), ("Sync word matched", s.get("sync_word_matched")),
        ]
        self.txt["Summary"].setPlainText("\n".join(f"{k:26s} {v if v not in (None, '') else '-'}" for k, v in fields))

        # bit views
        self.txt["Demodulated bits"].setPlainText(_bits_text(s.get("bitstream")))
        self.txt["De-interleaved bits"].setPlainText(_bits_text(s.get("deinterleaved_bits")))
        self.txt["FEC-corrected bits"].setPlainText(_bits_text(s.get("decoded_bits")))
        self.txt["Correlation"].setHtml(self._correlation_html(s))

        # text
        if "ascii_text" in s:
            body = (f"printable ratio: {s['printable_ratio']:.0%}   "
                    f"looks like valid text: {'yes' if s['text_looks_valid'] else 'NO'}\n\n{s['ascii_text']}")
        else:
            body = "(text stage did not run)"
        self.txt["Decoded text"].setPlainText(body)

    def _correlation_html(self, s):
        bits = s.get("decoded_bits")
        if bits is None:
            return "<i>(not available - earlier stage did not finish)</i>"
        bits = np.asarray(bits).astype(np.uint8).ravel()[:MAX_BITS_SHOWN]
        hs, he, ps, pe = (s.get(k) for k in ("header_start", "header_end", "payload_start", "payload_end"))
        if None in (hs, he, ps, pe):
            return "<i>No header/payload boundaries found.</i><br><br>" + html.escape("".join(map(str, bits)))
        seg = lambda a, b: "".join(map(str, bits[a:b]))     # noqa: E731
        return (
            f"<p>Sync word matched: <b>{html.escape(str(s.get('sync_word_matched') or '-'))}</b></p>"
            f"<p><span style='background:#ffe3b3'>header [{hs}:{he}]</span> "
            f"<span style='background:#c9f0d3'>payload [{ps}:{pe}]</span></p>"
            f"<p style='font-family:monospace'>{html.escape(seg(0, hs))}"
            f"<span style='background:#ffe3b3'>{html.escape(seg(hs, he))}</span>"
            f"{html.escape(seg(he, ps))}"
            f"<span style='background:#c9f0d3'>{html.escape(seg(ps, pe))}</span>"
            f"{html.escape(seg(pe, len(bits)))}</p>"
        )


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    panel = PipelinePanel()
    panel.resize(1000, 600)
    panel.show()
    sys.exit(app.exec_())