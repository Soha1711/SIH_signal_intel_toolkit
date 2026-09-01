"""
SIH 2026 - Signal Intelligence Toolkit
Main PyQt6 Dashboard

Features:
1. IQ / WAV file selection
2. Signal ingestion
3. Signal information
4. Waveform
5. FFT
6. PSD
7. Waterfall
8. Constellation
9. Signal parameter extraction
10. Export CSV / JSON
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
    QScrollArea,
)

from PyQt6.QtCore import Qt, QEvent


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

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------

    def setWidget(self, widget):
        """Install event filter recursively whenever a widget is set."""
        super().setWidget(widget)
        self._install_filter(widget)

    def _install_filter(self, widget):
        """Recursively install this scroll area as an event filter on
        every descendant so that wheel events are never swallowed."""
        widget.installEventFilter(self)
        for child in widget.findChildren(QWidget):
            child.installEventFilter(self)

    # ----------------------------------------------------------
    # Event filter — catches wheel events on ANY child widget
    # ----------------------------------------------------------

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            self._do_scroll(event)
            return True          # consume – don't let the child handle it
        return super().eventFilter(obj, event)

    # ----------------------------------------------------------
    # Wheel event on the scroll area itself
    # ----------------------------------------------------------

    def wheelEvent(self, event):
        self._do_scroll(event)
        event.accept()

    # ----------------------------------------------------------
    # Core scroll logic (handles both mouse wheel & touchpad)
    # ----------------------------------------------------------

    def _do_scroll(self, event):
        bar = self.verticalScrollBar()

        # Touchpad sends pixelDelta (precise pixel amounts)
        pixel_delta = event.pixelDelta().y()
        if pixel_delta != 0:
            bar.setValue(bar.value() - pixel_delta)
            return

        # Mouse wheel sends angleDelta (multiples of 120)
        angle_delta = event.angleDelta().y()
        if angle_delta != 0:
            # 120 units == one notch == ~3 lines (~60 px)
            bar.setValue(bar.value() - int(angle_delta / 120 * 60))

    # ----------------------------------------------------------
    # Keyboard scrolling
    # ----------------------------------------------------------

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

        # ----------------------------------------------------
        # Current data
        # ----------------------------------------------------

        self.current_signal = None
        self.current_result = None
        self.current_filepath = None

        # ----------------------------------------------------
        # Analysis widget
        # ----------------------------------------------------

        self.analysis_widget = SignalAnalysisWidget()

        # ----------------------------------------------------
        # Create GUI
        # ----------------------------------------------------

        self.setup_ui()


    # ========================================================
    # SETUP UI
    # ========================================================

    def setup_ui(self):

        # ----------------------------------------------------
        # Scroll area
        # ----------------------------------------------------

        scroll = SmoothScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        container = QWidget()

        main_layout = QVBoxLayout()

        main_layout.setSpacing(12)


        # ====================================================
        # TITLE
        # ====================================================

        title = QLabel(
            "SIH SIGNAL INTELLIGENCE TOOLKIT"
        )

        title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        title.setStyleSheet(
            """
            QLabel {
                font-size: 26px;
                font-weight: bold;
                padding: 15px;
            }
            """
        )

        main_layout.addWidget(title)


        # ====================================================
        # SIGNAL INPUT
        # ====================================================

        input_group = QGroupBox(
            "Signal Input"
        )

        input_layout = QHBoxLayout()

        self.file_path_label = QLabel(
            "No file selected"
        )

        self.file_path_label.setWordWrap(True)

        self.browse_button = QPushButton(
            "Browse..."
        )

        self.browse_button.clicked.connect(
            self.browse_file
        )

        input_layout.addWidget(
            self.file_path_label,
            1
        )

        input_layout.addWidget(
            self.browse_button
        )

        input_group.setLayout(
            input_layout
        )

        main_layout.addWidget(
            input_group
        )


        # ====================================================
        # IQ FILE PARAMETERS
        # ====================================================

        iq_group = QGroupBox(
            "IQ File Parameters"
        )

        iq_layout = QGridLayout()


        # ----------------------------------------------------
        # Sample rate
        # ----------------------------------------------------

        iq_layout.addWidget(
            QLabel("Sample Rate:"),
            0,
            0
        )

        self.sample_rate_input = QDoubleSpinBox()

        self.sample_rate_input.setRange(
            1,
            1_000_000_000
        )

        self.sample_rate_input.setValue(
            2_000_000
        )

        self.sample_rate_input.setDecimals(
            0
        )

        self.sample_rate_input.setSuffix(
            " Hz"
        )

        iq_layout.addWidget(
            self.sample_rate_input,
            0,
            1
        )


        # ----------------------------------------------------
        # Data type
        # ----------------------------------------------------

        iq_layout.addWidget(
            QLabel("Data Type:"),
            1,
            0
        )

        self.data_type_combo = QComboBox()

        self.data_type_combo.addItems(
            [
                "complex64",
                "complex128",
                "float32",
                "float64",
                "int16",
                "int8",
            ]
        )

        self.data_type_combo.setCurrentText(
            "float32"
        )

        iq_layout.addWidget(
            self.data_type_combo,
            1,
            1
        )

        iq_group.setLayout(
            iq_layout
        )

        main_layout.addWidget(
            iq_group
        )


        # ====================================================
        # LOAD SIGNAL
        # ====================================================

        self.load_button = QPushButton(
            "LOAD SIGNAL"
        )

        self.load_button.setMinimumHeight(
            45
        )

        self.load_button.setStyleSheet(
            """
            QPushButton {
                font-size: 16px;
                font-weight: bold;
            }
            """
        )

        self.load_button.clicked.connect(
            self.load_signal
        )

        main_layout.addWidget(
            self.load_button
        )


        # ====================================================
        # SIGNAL INFORMATION
        # ====================================================

        information_group = QGroupBox(
            "Signal Information"
        )

        information_layout = QGridLayout()


        # Format
        information_layout.addWidget(
            QLabel("Format:"),
            0,
            0
        )

        self.format_value = QLabel("-")

        information_layout.addWidget(
            self.format_value,
            0,
            1
        )


        # Sample rate
        information_layout.addWidget(
            QLabel("Sample Rate:"),
            1,
            0
        )

        self.info_sample_rate = QLabel("-")

        information_layout.addWidget(
            self.info_sample_rate,
            1,
            1
        )


        # Samples
        information_layout.addWidget(
            QLabel("Number of Samples:"),
            2,
            0
        )

        self.info_num_samples = QLabel("-")

        information_layout.addWidget(
            self.info_num_samples,
            2,
            1
        )


        # Duration
        information_layout.addWidget(
            QLabel("Duration:"),
            3,
            0
        )

        self.info_duration = QLabel("-")

        information_layout.addWidget(
            self.info_duration,
            3,
            1
        )


        # Data type
        information_layout.addWidget(
            QLabel("Data Type:"),
            4,
            0
        )

        self.info_data_type = QLabel("-")

        information_layout.addWidget(
            self.info_data_type,
            4,
            1
        )


        # Channels
        information_layout.addWidget(
            QLabel("Channels:"),
            5,
            0
        )

        self.info_channels = QLabel("-")

        information_layout.addWidget(
            self.info_channels,
            5,
            1
        )


        information_group.setLayout(
            information_layout
        )

        main_layout.addWidget(
            information_group
        )


        # ====================================================
        # SIGNAL PARAMETERS
        # ====================================================

        parameter_group = QGroupBox(
            "Signal Parameters"
        )

        parameter_layout = QGridLayout()


        # ----------------------------------------------------
        # Peak frequency
        # ----------------------------------------------------

        parameter_layout.addWidget(
            QLabel("Peak Frequency:"),
            0,
            0
        )

        self.peak_frequency_value = QLabel(
            "-"
        )

        self.peak_frequency_value.setStyleSheet(
            "font-weight: bold;"
        )

        parameter_layout.addWidget(
            self.peak_frequency_value,
            0,
            1
        )


        # ----------------------------------------------------
        # Bandwidth
        # ----------------------------------------------------

        parameter_layout.addWidget(
            QLabel("Bandwidth:"),
            1,
            0
        )

        self.bandwidth_value = QLabel(
            "-"
        )

        self.bandwidth_value.setStyleSheet(
            "font-weight: bold;"
        )

        parameter_layout.addWidget(
            self.bandwidth_value,
            1,
            1
        )


        # ----------------------------------------------------
        # SNR
        # ----------------------------------------------------

        parameter_layout.addWidget(
            QLabel("SNR:"),
            2,
            0
        )

        self.snr_value = QLabel(
            "-"
        )

        self.snr_value.setStyleSheet(
            "font-weight: bold;"
        )

        parameter_layout.addWidget(
            self.snr_value,
            2,
            1
        )


        # ----------------------------------------------------
        # Modulation
        # ----------------------------------------------------

        parameter_layout.addWidget(
            QLabel("Modulation:"),
            3,
            0
        )

        self.modulation_value = QLabel(
            "-"
        )

        self.modulation_value.setStyleSheet(
            "font-weight: bold;"
        )

        parameter_layout.addWidget(
            self.modulation_value,
            3,
            1
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        parameter_layout.addWidget(
            QLabel("Confidence:"),
            4,
            0
        )

        self.confidence_value = QLabel(
            "-"
        )

        parameter_layout.addWidget(
            self.confidence_value,
            4,
            1
        )


        parameter_group.setLayout(
            parameter_layout
        )

        main_layout.addWidget(
            parameter_group
        )


        # ====================================================
        # STATUS
        # ====================================================

        self.status_label = QLabel(
            "Status: Ready"
        )

        self.status_label.setStyleSheet(
            """
            QLabel {
                padding: 8px;
                font-weight: bold;
            }
            """
        )

        main_layout.addWidget(
            self.status_label
        )


        # ====================================================
        # SIGNAL ANALYSIS
        # ====================================================

        main_layout.addWidget(
            self.analysis_widget
        )


        # ====================================================
        # EXPORT SECTION
        # ====================================================

        export_group = QGroupBox(
            "Export Results"
        )

        export_layout = QHBoxLayout()


        self.export_csv_button = QPushButton(
            "Export CSV"
        )

        self.export_json_button = QPushButton(
            "Export JSON"
        )

        self.export_png_button = QPushButton(
            "Save Plot"
        )


        self.export_csv_button.clicked.connect(
            self.export_csv
        )

        self.export_json_button.clicked.connect(
            self.export_json
        )

        self.export_png_button.clicked.connect(
            self.export_plot
        )


        export_layout.addWidget(
            self.export_csv_button
        )

        export_layout.addWidget(
            self.export_json_button
        )

        export_layout.addWidget(
            self.export_png_button
        )


        export_group.setLayout(
            export_layout
        )

        main_layout.addWidget(
            export_group
        )


        # ====================================================
        # CONTAINER
        # ====================================================

        container.setLayout(
            main_layout
        )

        scroll.setWidget(
            container
        )


        outer_layout = QVBoxLayout()

        outer_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        outer_layout.addWidget(
            scroll
        )

        self.setLayout(
            outer_layout
        )


        # ====================================================
        # WINDOW
        # ====================================================

        self.setWindowTitle(
            "SIH Signal Intelligence Toolkit"
        )

        self.resize(
            1200,
            900
        )


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

        self.file_path_label.setText(
            file_path
        )

        self.status_label.setText(
            "Status: File selected. Click LOAD SIGNAL."
        )


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


        self.status_label.setText(
            "Status: Loading and analyzing signal..."
        )

        QApplication.processEvents()


        try:

            # ------------------------------------------------
            # Sample rate
            # ------------------------------------------------

            sample_rate = float(
                self.sample_rate_input.value()
            )


            # ------------------------------------------------
            # Data type
            # ------------------------------------------------

            iq_dtype = (
                self.data_type_combo.currentText()
            )


            # ------------------------------------------------
            # Load file
            # ------------------------------------------------

            result = load_signal_file(
                self.current_filepath,
                sample_rate=sample_rate,
                iq_dtype=iq_dtype
            )


            if result is None:
                raise ValueError(
                    "Signal loader returned no result."
                )


            if "samples" not in result:
                raise KeyError(
                    "Loader result does not contain samples."
                )


            if "sample_rate" not in result:
                raise KeyError(
                    "Loader result does not contain sample_rate."
                )


            # ------------------------------------------------
            # Save signal
            # ------------------------------------------------

            self.current_signal = np.asarray(
                result["samples"]
            )


            # ------------------------------------------------
            # Signal information
            # ------------------------------------------------

            self.display_signal_information(
                result
            )


            # ------------------------------------------------
            # Full signal analysis
            # ------------------------------------------------

            self.current_result = analyze_signal(
                result
            )


            if (
                self.current_result.get(
                    "analysis_status"
                )
                != "success"
            ):

                raise ValueError(
                    self.current_result.get(
                        "error",
                        "Signal analysis failed."
                    )
                )


            # ------------------------------------------------
            # Display graphs
            # ------------------------------------------------

            self.analysis_widget.display_analysis(
                self.current_result
            )


            # ------------------------------------------------
            # Display parameters
            # ------------------------------------------------

            self.display_parameters(
                self.current_result
            )


            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            self.status_label.setText(
                "Status: ✓ Signal loaded and analyzed successfully."
            )


        except Exception as error:

            self.current_signal = None
            self.current_result = None

            self.analysis_widget.clear_plots()

            self.status_label.setText(
                "Status: ✗ Error - " + str(error)
            )

            QMessageBox.critical(
                self,
                "Signal Analysis Error",
                str(error)
            )


    # ========================================================
    # DISPLAY SIGNAL INFORMATION
    # ========================================================

    def display_signal_information(
        self,
        result
    ):

        samples = result.get(
            "samples"
        )

        sample_rate = result.get(
            "sample_rate"
        )


        # ----------------------------------------------------
        # Format
        # ----------------------------------------------------

        source_format = result.get(
            "source_format",
            "-"
        )

        self.format_value.setText(
            str(source_format).upper()
        )


        # ----------------------------------------------------
        # Sample rate
        # ----------------------------------------------------

        if sample_rate is not None:

            self.info_sample_rate.setText(
                f"{float(sample_rate):,.2f} Hz"
            )


        # ----------------------------------------------------
        # Number samples
        # ----------------------------------------------------

        if samples is not None:

            self.info_num_samples.setText(
                f"{len(samples):,}"
            )


        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        if (
            samples is not None
            and sample_rate is not None
            and float(sample_rate) > 0
        ):

            duration = (
                len(samples)
                / float(sample_rate)
            )

            self.info_duration.setText(
                f"{duration:.6f} sec"
            )


        # ----------------------------------------------------
        # Data type
        # ----------------------------------------------------

        if samples is not None:

            self.info_data_type.setText(
                str(samples.dtype)
            )


        # ----------------------------------------------------
        # Channels
        # ----------------------------------------------------

        channels = result.get(
            "channels"
        )

        if channels is not None:

            self.info_channels.setText(
                str(channels)
            )

        elif samples is not None:

            if np.iscomplexobj(samples):

                self.info_channels.setText(
                    "2 (I + Q)"
                )

            else:

                self.info_channels.setText(
                    "1"
                )


    # ========================================================
    # PARAMETER EXTRACTION
    # ========================================================

    def display_parameters(
        self,
        result
    ):

        # ----------------------------------------------------
        # Peak frequency
        # ----------------------------------------------------

        peak_frequency = result.get(
            "peak_frequency"
        )

        if peak_frequency is not None:

            self.peak_frequency_value.setText(
                self.format_frequency(
                    peak_frequency
                )
            )


        # ----------------------------------------------------
        # Bandwidth
        # ----------------------------------------------------

        bandwidth = self.calculate_bandwidth(
            result
        )

        self.bandwidth_value.setText(
            self.format_frequency(
                bandwidth
            )
        )


        # ----------------------------------------------------
        # SNR
        # ----------------------------------------------------

        snr = self.calculate_snr()

        self.snr_value.setText(
            f"{snr:.2f} dB"
        )


        # ----------------------------------------------------
        # Modulation
        # ----------------------------------------------------

        modulation = result.get(
            "detected_modulation",
            "Unknown"
        )

        self.modulation_value.setText(
            str(modulation)
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = result.get(
            "modulation_confidence",
            0
        )

        try:

            self.confidence_value.setText(
                f"{float(confidence) * 100:.1f}%"
            )

        except Exception:

            self.confidence_value.setText(
                "-"
            )


    # ========================================================
    # FREQUENCY FORMAT
    # ========================================================

    def format_frequency(
        self,
        frequency
    ):

        frequency = abs(
            float(frequency)
        )

        if frequency >= 1_000_000:

            return f"{frequency / 1_000_000:.3f} MHz"

        elif frequency >= 1_000:

            return f"{frequency / 1_000:.3f} kHz"

        else:

            return f"{frequency:.2f} Hz"


    # ========================================================
    # BANDWIDTH
    # ========================================================

    def calculate_bandwidth(
        self,
        result
    ):

        frequencies = result.get(
            "fft_frequency"
        )

        magnitude = result.get(
            "fft_magnitude_db"
        )


        if frequencies is None or magnitude is None:

            return 0


        frequencies = np.asarray(
            frequencies
        )

        magnitude = np.asarray(
            magnitude
        )


        if len(magnitude) == 0:

            return 0


        # ----------------------------------------------------
        # -3 dB bandwidth
        # ----------------------------------------------------

        peak = np.max(
            magnitude
        )

        threshold = peak - 3.0


        indices = np.where(
            magnitude >= threshold
        )[0]


        if len(indices) < 2:

            return 0


        bandwidth = (
            frequencies[indices[-1]]
            -
            frequencies[indices[0]]
        )


        return abs(
            bandwidth
        )


    # ========================================================
    # SNR
    # ========================================================

    def calculate_snr(
        self
    ):

        if self.current_signal is None:

            return 0


        signal = np.asarray(
            self.current_signal
        )


        if len(signal) == 0:

            return 0


        # ----------------------------------------------------
        # Signal power
        # ----------------------------------------------------

        signal_power = np.mean(
            np.abs(signal) ** 2
        )


        # ----------------------------------------------------
        # Estimate noise
        # ----------------------------------------------------

        spectrum = np.abs(
            np.fft.fft(signal)
        )


        if len(spectrum) == 0:

            return 0


        peak_index = np.argmax(
            spectrum
        )


        # Remove strongest component
        spectrum_copy = spectrum.copy()

        start = max(
            0,
            peak_index - 3
        )

        end = min(
            len(spectrum),
            peak_index + 4
        )

        spectrum_copy[
            start:end
        ] = 0


        noise_power = np.mean(
            spectrum_copy ** 2
        )


        if noise_power <= 0:

            return 0


        snr = 10 * np.log10(
            signal_power
            /
            noise_power
        )


        return float(
            snr
        )


    # ========================================================
    # EXPORT JSON
    # ========================================================

    def export_json(self):

        if self.current_result is None:

            QMessageBox.warning(
                self,
                "No Data",
                "Load and analyze a signal first."
            )

            return


        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON Report",
            "signal_report.json",
            "JSON Files (*.json)"
        )


        if not file_path:

            return


        result = self.current_result


        data = {

            "file": self.current_filepath,

            "format":
                self.format_value.text(),

            "sample_rate":
                self.info_sample_rate.text(),

            "number_of_samples":
                self.info_num_samples.text(),

            "duration":
                self.info_duration.text(),

            "data_type":
                self.info_data_type.text(),

            "channels":
                self.info_channels.text(),

            "peak_frequency":
                self.peak_frequency_value.text(),

            "bandwidth":
                self.bandwidth_value.text(),

            "snr":
                self.snr_value.text(),

            "modulation":
                self.modulation_value.text(),

            "confidence":
                self.confidence_value.text()
        }


        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4
            )


        QMessageBox.information(
            self,
            "Export Successful",
            "JSON report saved successfully."
        )


    # ========================================================
    # EXPORT CSV
    # ========================================================

    def export_csv(self):

        if self.current_result is None:

            QMessageBox.warning(
                self,
                "No Data",
                "Load and analyze a signal first."
            )

            return


        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV Report",
            "signal_report.csv",
            "CSV Files (*.csv)"
        )


        if not file_path:

            return


        rows = [

            ["Parameter", "Value"],

            [
                "File",
                self.current_filepath
            ],

            [
                "Format",
                self.format_value.text()
            ],

            [
                "Sample Rate",
                self.info_sample_rate.text()
            ],

            [
                "Number of Samples",
                self.info_num_samples.text()
            ],

            [
                "Duration",
                self.info_duration.text()
            ],

            [
                "Data Type",
                self.info_data_type.text()
            ],

            [
                "Channels",
                self.info_channels.text()
            ],

            [
                "Peak Frequency",
                self.peak_frequency_value.text()
            ],

            [
                "Bandwidth",
                self.bandwidth_value.text()
            ],

            [
                "SNR",
                self.snr_value.text()
            ],

            [
                "Modulation",
                self.modulation_value.text()
            ],

            [
                "Confidence",
                self.confidence_value.text()
            ]
        ]


        with open(
            file_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerows(
                rows
            )


        QMessageBox.information(
            self,
            "Export Successful",
            "CSV report saved successfully."
        )


    # ========================================================
    # SAVE PLOT
    # ========================================================

    def export_plot(self):

        if self.current_result is None:

            QMessageBox.warning(
                self,
                "No Data",
                "Load a signal first."
            )

            return


        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "signal_analysis.png",
            "PNG Files (*.png)"
        )


        if not file_path:

            return


        # Save the main analysis figure
        self.analysis_widget.row1_figure.savefig(
            file_path,
            dpi=200,
            bbox_inches="tight"
        )


        QMessageBox.information(
            self,
            "Export Successful",
            "Plot saved successfully."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    app = QApplication(
        sys.argv
    )

    window = SignalIntelligenceGUI()

    window.show()

    sys.exit(
        app.exec()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
