"""
SIH 2026 - Signal Intelligence Toolkit
Main PyQt6 Dashboard

Layout:
  Top panel  — title, signal input, IQ params, LOAD button,
                signal info, signal parameters, status
  Bottom panel — tabbed plots (Waveform / FFT / PSD / Waterfall / Constellation)
  Footer     — Export CSV | Export JSON | Save Plot
"""

import sys
import csv
import json
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QFileDialog,
    QDoubleSpinBox,
    QComboBox,
    QMessageBox,
    QSplitter,
    QScrollArea,
    QFrame,
)

from PyQt6.QtCore import Qt, QEvent


# ============================================================
# DARK THEME STYLESHEET
# ============================================================

DARK_STYLE = """
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    font-size: 12px;
    color: #aaaaaa;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}

QLabel {
    color: #cccccc;
}

QLabel#value_label {
    color: #ffffff;
    font-weight: bold;
}

QPushButton {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #4a4a4a;
    border-radius: 3px;
    padding: 6px 14px;
}

QPushButton:hover {
    background-color: #3a3a3a;
    border-color: #666;
}

QPushButton:pressed {
    background-color: #252525;
}

QPushButton#load_btn {
    background-color: #2a2a2a;
    color: #ffffff;
    font-size: 15px;
    font-weight: bold;
    border: 1px solid #555;
    padding: 10px;
    border-radius: 3px;
}

QPushButton#load_btn:hover {
    background-color: #383838;
    border-color: #888;
}

QDoubleSpinBox, QComboBox {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #4a4a4a;
    border-radius: 3px;
    padding: 4px 8px;
    min-height: 26px;
}

QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #3a3a3a;
    width: 18px;
}

QComboBox::drop-down {
    border: none;
    background-color: #3a3a3a;
    width: 22px;
}

QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    color: #e0e0e0;
    selection-background-color: #404040;
}

QTabWidget::pane {
    border: 1px solid #3c3c3c;
    background-color: #1e1e1e;
}

QTabBar::tab {
    background-color: #2a2a2a;
    color: #aaaaaa;
    border: 1px solid #3c3c3c;
    border-bottom: none;
    padding: 6px 18px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #ffffff;
    border-top: 2px solid #888;
}

QTabBar::tab:hover {
    background-color: #333333;
    color: #dddddd;
}

QScrollArea {
    border: none;
    background-color: #1e1e1e;
}

QScrollBar:vertical {
    background-color: #2a2a2a;
    width: 10px;
}

QScrollBar::handle:vertical {
    background-color: #555;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QSplitter::handle {
    background-color: #3c3c3c;
    height: 2px;
}

QFrame#separator {
    background-color: #3c3c3c;
    max-height: 1px;
}
"""


# ============================================================
# SMOOTH SCROLL AREA (mouse wheel + touchpad + keyboard)
# ============================================================

class SmoothScrollArea(QScrollArea):
    """
    QScrollArea with full scroll support:
      - Mouse wheel
      - Laptop touchpad (smooth / pixel-delta scrolling)
      - Keyboard: Up/Down/PageUp/PageDown/Home/End
    Works even when child widgets (e.g. matplotlib canvases) would
    normally consume the wheel event themselves.
    """

    KEY_STEP  = 40    # px per arrow-key press
    PAGE_STEP = 300   # px per Page Up / Page Down

    def setWidget(self, widget):
        super().setWidget(widget)
        self._install_filter(widget)

    def _install_filter(self, widget):
        widget.installEventFilter(self)
        for child in widget.findChildren(QWidget):
            child.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            self._do_scroll(event)
            return True
        return super().eventFilter(obj, event)

    def wheelEvent(self, event):
        self._do_scroll(event)
        event.accept()

    def _do_scroll(self, event):
        bar = self.verticalScrollBar()
        pixel_delta = event.pixelDelta().y()
        if pixel_delta != 0:
            bar.setValue(bar.value() - pixel_delta)
            return
        angle_delta = event.angleDelta().y()
        if angle_delta != 0:
            bar.setValue(bar.value() - int(angle_delta / 120 * 60))

    def keyPressEvent(self, event):
        bar = self.verticalScrollBar()
        key = event.key()
        if key == Qt.Key.Key_Up:
            bar.setValue(bar.value() - self.KEY_STEP)
        elif key == Qt.Key.Key_Down:
            bar.setValue(bar.value() + self.KEY_STEP)
        elif key == Qt.Key.Key_PageUp:
            bar.setValue(bar.value() - self.PAGE_STEP)
        elif key == Qt.Key.Key_PageDown:
            bar.setValue(bar.value() + self.PAGE_STEP)
        elif key == Qt.Key.Key_Home:
            bar.setValue(bar.minimum())
        elif key == Qt.Key.Key_End:
            bar.setValue(bar.maximum())
        else:
            super().keyPressEvent(event)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT MODULES
# ============================================================

from ingestion.iq_wav_reader import load_signal_file
from param_estimation.signal_analysis import analyze_signal
from gui.signal_analysis_widget import SignalAnalysisWidget


# ============================================================
# MAIN GUI
# ============================================================

class SignalIntelligenceGUI(QWidget):

    def __init__(self):
        super().__init__()

        self.current_signal   = None
        self.current_result   = None
        self.current_filepath = None

        self.analysis_widget = SignalAnalysisWidget()

        self.setup_ui()
        self.apply_dark_theme()


    # ========================================================
    # APPLY THEME
    # ========================================================

    def apply_dark_theme(self):
        self.setStyleSheet(DARK_STYLE)

        # Dark background for matplotlib figures
        for fig in [
            self.analysis_widget.waveform_figure,
            self.analysis_widget.fft_figure,
            self.analysis_widget.psd_figure,
            self.analysis_widget.waterfall_figure,
            self.analysis_widget.constellation_figure,
        ]:
            fig.patch.set_facecolor("#1e1e1e")
            for ax in fig.get_axes():
                ax.set_facecolor("#1e1e1e")
                ax.tick_params(colors="#aaaaaa")
                ax.xaxis.label.set_color("#aaaaaa")
                ax.yaxis.label.set_color("#aaaaaa")
                ax.title.set_color("#dddddd")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#444444")


    # ========================================================
    # SETUP UI
    # ========================================================

    def setup_ui(self):

        # ----------------------------------------------------
        # Root splitter  (top controls | bottom plots)
        # ----------------------------------------------------

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(4)

        # ====================================================
        # TOP PANEL  (scrollable control area)
        # ====================================================

        top_scroll = SmoothScrollArea()
        top_scroll.setWidgetResizable(True)
        top_scroll.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        top_scroll.setFrameShape(QFrame.Shape.NoFrame)

        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setSpacing(8)
        top_layout.setContentsMargins(10, 8, 10, 8)

        # ------------------------------------------------
        # TITLE
        # ------------------------------------------------

        title = QLabel("SIH SIGNAL INTELLIGENCE TOOLKIT")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #ffffff;"
            "padding: 10px 0 6px 0; letter-spacing: 1px;"
        )
        top_layout.addWidget(title)

        # ------------------------------------------------
        # SIGNAL INPUT
        # ------------------------------------------------

        input_group = QGroupBox("Signal Input")
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(8, 4, 8, 4)

        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setStyleSheet("color: #aaaaaa;")

        self.browse_button = QPushButton("Browse...")
        self.browse_button.setFixedWidth(90)
        self.browse_button.clicked.connect(self.browse_file)

        input_layout.addWidget(self.file_path_label, 1)
        input_layout.addWidget(self.browse_button)
        input_group.setLayout(input_layout)
        top_layout.addWidget(input_group)

        # ------------------------------------------------
        # IQ FILE PARAMETERS
        # ------------------------------------------------

        iq_group = QGroupBox("IQ File Parameters")
        iq_layout = QGridLayout()
        iq_layout.setContentsMargins(8, 4, 8, 4)
        iq_layout.setColumnStretch(1, 1)

        # Sample rate
        iq_layout.addWidget(QLabel("Sample Rate:"), 0, 0)

        self.sample_rate_input = QDoubleSpinBox()
        self.sample_rate_input.setRange(1, 1_000_000_000)
        self.sample_rate_input.setValue(2_000_000)
        self.sample_rate_input.setDecimals(0)
        self.sample_rate_input.setSuffix(" Hz")
        iq_layout.addWidget(self.sample_rate_input, 0, 1)

        # Data type
        iq_layout.addWidget(QLabel("Data Type:"), 1, 0)

        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems([
            "complex64",
            "complex128",
            "float32",
            "float64",
            "int16",
            "int8",
        ])
        self.data_type_combo.setCurrentText("float32")
        iq_layout.addWidget(self.data_type_combo, 1, 1)

        iq_group.setLayout(iq_layout)
        top_layout.addWidget(iq_group)

        # ------------------------------------------------
        # LOAD SIGNAL BUTTON
        # ------------------------------------------------

        self.load_button = QPushButton("LOAD SIGNAL")
        self.load_button.setObjectName("load_btn")
        self.load_button.setMinimumHeight(42)
        self.load_button.clicked.connect(self.load_signal)
        top_layout.addWidget(self.load_button)

        # ------------------------------------------------
        # SIGNAL INFORMATION
        # ------------------------------------------------

        info_group = QGroupBox("Signal Information")
        info_layout = QGridLayout()
        info_layout.setContentsMargins(8, 4, 8, 4)
        info_layout.setColumnStretch(1, 1)

        def _info_row(layout, row, label_text):
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #999;")
            val = QLabel("-")
            val.setObjectName("value_label")
            layout.addWidget(lbl, row, 0)
            layout.addWidget(val, row, 1)
            return val

        self.format_value       = _info_row(info_layout, 0, "Format:")
        self.info_sample_rate   = _info_row(info_layout, 1, "Sample Rate:")
        self.info_num_samples   = _info_row(info_layout, 2, "Number of Samples:")
        self.info_duration      = _info_row(info_layout, 3, "Duration:")
        self.info_data_type     = _info_row(info_layout, 4, "Data Type:")
        self.info_channels      = _info_row(info_layout, 5, "Channels:")

        info_group.setLayout(info_layout)
        top_layout.addWidget(info_group)

        # ------------------------------------------------
        # SIGNAL PARAMETERS
        # ------------------------------------------------

        param_group = QGroupBox("Signal Parameters")
        param_layout = QGridLayout()
        param_layout.setContentsMargins(8, 4, 8, 4)
        param_layout.setColumnStretch(1, 1)

        self.peak_frequency_value = _info_row(param_layout, 0, "Peak Frequency:")
        self.bandwidth_value      = _info_row(param_layout, 1, "Bandwidth:")
        self.snr_value            = _info_row(param_layout, 2, "SNR:")
        self.modulation_value     = _info_row(param_layout, 3, "Modulation:")
        self.confidence_value     = _info_row(param_layout, 4, "Confidence:")

        # Bold for parameter values
        for val in [
            self.peak_frequency_value,
            self.bandwidth_value,
            self.snr_value,
            self.modulation_value,
        ]:
            val.setStyleSheet("font-weight: bold; color: #ffffff;")

        param_group.setLayout(param_layout)
        top_layout.addWidget(param_group)

        # ------------------------------------------------
        # STATUS
        # ------------------------------------------------

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet(
            "padding: 6px 0; font-weight: bold; color: #cccccc;"
        )
        top_layout.addWidget(self.status_label)

        top_layout.addStretch(1)
        top_scroll.setWidget(top_container)

        # ====================================================
        # BOTTOM PANEL  (tabbed plots + export buttons)
        # ====================================================

        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        # Analysis widget (QTabWidget with 5 tabs)
        bottom_layout.addWidget(self.analysis_widget, 1)

        # Export buttons row
        export_layout = QHBoxLayout()
        export_layout.setContentsMargins(8, 4, 8, 6)
        export_layout.setSpacing(6)

        self.export_csv_button  = QPushButton("Export CSV")
        self.export_json_button = QPushButton("Export JSON")
        self.export_png_button  = QPushButton("Save Plot")

        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_json_button.clicked.connect(self.export_json)
        self.export_png_button.clicked.connect(self.export_plot)

        for btn in [
            self.export_csv_button,
            self.export_json_button,
            self.export_png_button,
        ]:
            btn.setMinimumHeight(32)
            export_layout.addWidget(btn)

        bottom_layout.addLayout(export_layout)

        # ====================================================
        # SPLITTER ASSEMBLY
        # ====================================================

        splitter.addWidget(top_scroll)
        splitter.addWidget(bottom_widget)

        # Default split: 35% top, 65% bottom
        splitter.setSizes([350, 650])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Root layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(splitter)

        # ====================================================
        # WINDOW
        # ====================================================

        self.setWindowTitle("SIH Signal Intelligence Toolkit")
        self.resize(1200, 900)


    # ========================================================
    # BROWSE FILE
    # ========================================================

    def browse_file(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select IQ or WAV File",
            "",
            "Signal Files (*.iq *.IQ *.wav *.WAV);;"
            "IQ Files (*.iq *.IQ);;"
            "WAV Files (*.wav *.WAV);;"
            "All Files (*)"
        )

        if not file_path:
            return

        self.current_filepath = file_path
        self.file_path_label.setText(file_path)
        self.file_path_label.setStyleSheet("color: #cccccc;")
        self.status_label.setText("Status: File selected. Click LOAD SIGNAL.")


    # ========================================================
    # LOAD SIGNAL
    # ========================================================

    def load_signal(self):

        if not self.current_filepath:
            QMessageBox.warning(
                self,
                "No File Selected",
                "Please select an IQ or WAV file first."
            )
            return

        self.status_label.setText("Status: Loading and analyzing signal...")
        QApplication.processEvents()

        try:
            sample_rate = float(self.sample_rate_input.value())
            iq_dtype    = self.data_type_combo.currentText()

            result = load_signal_file(
                self.current_filepath,
                sample_rate=sample_rate,
                iq_dtype=iq_dtype
            )

            if result is None:
                raise ValueError("Signal loader returned no result.")
            if "samples" not in result:
                raise KeyError("Loader result does not contain samples.")
            if "sample_rate" not in result:
                raise KeyError("Loader result does not contain sample_rate.")

            self.current_signal = np.asarray(result["samples"])

            self.display_signal_information(result)

            self.current_result = analyze_signal(result)

            if self.current_result.get("analysis_status") != "success":
                raise ValueError(
                    self.current_result.get("error", "Signal analysis failed.")
                )

            self.analysis_widget.display_analysis(self.current_result)
            self._refresh_plot_theme()

            self.display_parameters(self.current_result)

            self.status_label.setText(
                "Status: ✓ Signal loaded and analyzed successfully."
            )
            self.status_label.setStyleSheet(
                "padding: 6px 0; font-weight: bold; color: #66bb6a;"
            )

        except Exception as error:
            self.current_signal = None
            self.current_result = None
            self.analysis_widget.clear_plots()
            self.status_label.setText("Status: ✗ Error — " + str(error))
            self.status_label.setStyleSheet(
                "padding: 6px 0; font-weight: bold; color: #ef5350;"
            )
            QMessageBox.critical(self, "Signal Analysis Error", str(error))


    # ========================================================
    # RE-APPLY DARK THEME TO NEWLY DRAWN FIGURES
    # ========================================================

    def _refresh_plot_theme(self):
        for fig in [
            self.analysis_widget.waveform_figure,
            self.analysis_widget.fft_figure,
            self.analysis_widget.psd_figure,
            self.analysis_widget.waterfall_figure,
            self.analysis_widget.constellation_figure,
        ]:
            fig.patch.set_facecolor("#1e1e1e")
            for ax in fig.get_axes():
                ax.set_facecolor("#252525")
                ax.tick_params(colors="#aaaaaa")
                ax.xaxis.label.set_color("#aaaaaa")
                ax.yaxis.label.set_color("#aaaaaa")
                ax.title.set_color("#dddddd")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#444444")
                ax.grid(True, color="#333333", linewidth=0.6, alpha=0.8)

        # Redraw all canvases
        for canvas in [
            self.analysis_widget.waveform_canvas,
            self.analysis_widget.fft_canvas,
            self.analysis_widget.psd_canvas,
            self.analysis_widget.waterfall_canvas,
            self.analysis_widget.constellation_canvas,
        ]:
            canvas.draw()


    # ========================================================
    # DISPLAY SIGNAL INFORMATION
    # ========================================================

    def display_signal_information(self, result):

        samples     = result.get("samples")
        sample_rate = result.get("sample_rate")

        self.format_value.setText(
            str(result.get("source_format", "-")).upper()
        )

        if sample_rate is not None:
            self.info_sample_rate.setText(f"{float(sample_rate):,.2f} Hz")

        if samples is not None:
            self.info_num_samples.setText(f"{len(samples):,}")

        if samples is not None and sample_rate is not None and float(sample_rate) > 0:
            duration = len(samples) / float(sample_rate)
            self.info_duration.setText(f"{duration:.6f} sec")

        if samples is not None:
            self.info_data_type.setText(str(samples.dtype))

        channels = result.get("channels")
        if channels is not None:
            self.info_channels.setText(str(channels))
        elif samples is not None:
            self.info_channels.setText(
                "2 (I + Q)" if np.iscomplexobj(samples) else "1"
            )


    # ========================================================
    # DISPLAY PARAMETERS
    # ========================================================

    def display_parameters(self, result):

        peak_frequency = result.get("peak_frequency")
        if peak_frequency is not None:
            self.peak_frequency_value.setText(
                self.format_frequency(peak_frequency)
            )

        bandwidth = self.calculate_bandwidth(result)
        self.bandwidth_value.setText(self.format_frequency(bandwidth))

        snr = self.calculate_snr()
        self.snr_value.setText(f"{snr:.2f} dB")

        modulation = result.get("detected_modulation", "Unknown")
        self.modulation_value.setText(str(modulation))

        confidence = result.get("modulation_confidence", 0)
        try:
            self.confidence_value.setText(f"{float(confidence) * 100:.1f}%")
        except Exception:
            self.confidence_value.setText("-")


    # ========================================================
    # FREQUENCY FORMAT
    # ========================================================

    def format_frequency(self, frequency):
        frequency = abs(float(frequency))
        if frequency >= 1_000_000:
            return f"{frequency / 1_000_000:.3f} MHz"
        elif frequency >= 1_000:
            return f"{frequency / 1_000:.3f} kHz"
        else:
            return f"{frequency:.2f} Hz"


    # ========================================================
    # BANDWIDTH
    # ========================================================

    def calculate_bandwidth(self, result):
        frequencies = result.get("fft_frequency")
        magnitude   = result.get("fft_magnitude_db")

        if frequencies is None or magnitude is None:
            return 0

        frequencies = np.asarray(frequencies)
        magnitude   = np.asarray(magnitude)

        if len(magnitude) == 0:
            return 0

        peak      = np.max(magnitude)
        threshold = peak - 3.0
        indices   = np.where(magnitude >= threshold)[0]

        if len(indices) < 2:
            return 0

        return abs(frequencies[indices[-1]] - frequencies[indices[0]])


    # ========================================================
    # SNR
    # ========================================================

    def calculate_snr(self):
        if self.current_signal is None:
            return 0

        signal = np.asarray(self.current_signal)
        if len(signal) == 0:
            return 0

        signal_power = np.mean(np.abs(signal) ** 2)

        spectrum     = np.abs(np.fft.fft(signal))
        if len(spectrum) == 0:
            return 0

        peak_index   = np.argmax(spectrum)
        spectrum_copy = spectrum.copy()
        start = max(0, peak_index - 3)
        end   = min(len(spectrum), peak_index + 4)
        spectrum_copy[start:end] = 0

        noise_power = np.mean(spectrum_copy ** 2)
        if noise_power <= 0:
            return 0

        return float(10 * np.log10(signal_power / noise_power))


    # ========================================================
    # EXPORT JSON
    # ========================================================

    def export_json(self):
        if self.current_result is None:
            QMessageBox.warning(self, "No Data", "Load and analyze a signal first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON Report", "signal_report.json", "JSON Files (*.json)"
        )
        if not file_path:
            return

        data = {
            "file":              self.current_filepath,
            "format":            self.format_value.text(),
            "sample_rate":       self.info_sample_rate.text(),
            "number_of_samples": self.info_num_samples.text(),
            "duration":          self.info_duration.text(),
            "data_type":         self.info_data_type.text(),
            "channels":          self.info_channels.text(),
            "peak_frequency":    self.peak_frequency_value.text(),
            "bandwidth":         self.bandwidth_value.text(),
            "snr":               self.snr_value.text(),
            "modulation":        self.modulation_value.text(),
            "confidence":        self.confidence_value.text(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        QMessageBox.information(self, "Export Successful", "JSON report saved.")


    # ========================================================
    # EXPORT CSV
    # ========================================================

    def export_csv(self):
        if self.current_result is None:
            QMessageBox.warning(self, "No Data", "Load and analyze a signal first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV Report", "signal_report.csv", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        rows = [
            ["Parameter",       "Value"],
            ["File",            self.current_filepath],
            ["Format",          self.format_value.text()],
            ["Sample Rate",     self.info_sample_rate.text()],
            ["Num Samples",     self.info_num_samples.text()],
            ["Duration",        self.info_duration.text()],
            ["Data Type",       self.info_data_type.text()],
            ["Channels",        self.info_channels.text()],
            ["Peak Frequency",  self.peak_frequency_value.text()],
            ["Bandwidth",       self.bandwidth_value.text()],
            ["SNR",             self.snr_value.text()],
            ["Modulation",      self.modulation_value.text()],
            ["Confidence",      self.confidence_value.text()],
        ]

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

        QMessageBox.information(self, "Export Successful", "CSV report saved.")


    # ========================================================
    # SAVE PLOT
    # ========================================================

    def export_plot(self):
        if self.current_result is None:
            QMessageBox.warning(self, "No Data", "Load a signal first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "signal_plot.png", "PNG Files (*.png)"
        )
        if not file_path:
            return

        # Save whichever tab is currently visible
        idx = self.analysis_widget.tabs.currentIndex()
        figures = [
            self.analysis_widget.waveform_figure,
            self.analysis_widget.fft_figure,
            self.analysis_widget.psd_figure,
            self.analysis_widget.waterfall_figure,
            self.analysis_widget.constellation_figure,
        ]
        figures[idx].savefig(file_path, dpi=200, bbox_inches="tight",
                             facecolor="#1e1e1e")

        QMessageBox.information(self, "Export Successful", "Plot saved.")


# ============================================================
# MAIN
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SignalIntelligenceGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
