"""
SIH 2026 - Signal Intelligence Toolkit

Main GUI for:
    1. IQ / WAV file selection
    2. Signal ingestion
    3. Signal information display
    4. Waveform analysis
    5. FFT analysis
    6. PSD analysis

The ingestion module is responsible for reading IQ/WAV files.

The signal analysis module receives:
    result["samples"]
    result["sample_rate"]

and generates:
    Waveform
    FFT
    PSD
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QFileDialog,
    QDoubleSpinBox,
    QComboBox,
    QGridLayout,
    QMessageBox,
    QScrollArea,
    QMainWindow,
)

from PyQt6.QtCore import Qt


# ============================================================
# MAKE PROJECT ROOT AVAILABLE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT TEAM INGESTION MODULE
# ============================================================

from ingestion.iq_wav_reader import load_signal_file


# ============================================================
# IMPORT SIGNAL ANALYSIS MODULE
# ============================================================

from param_estimation.signal_analysis import analyze_signal


# ============================================================
# IMPORT GUI GRAPH WIDGET
# ============================================================

from gui.signal_analysis_widget import SignalAnalysisWidget


# ============================================================
# MAIN WINDOW
# ============================================================

class SignalIntelligenceGUI(QWidget):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Store current signal information
        # ----------------------------------------------------

        self.current_signal = None
        self.current_result = None
        self.current_filepath = None

        # ----------------------------------------------------
        # Create signal analysis widget
        # ----------------------------------------------------

        self.analysis_widget = SignalAnalysisWidget()

        # ----------------------------------------------------
        # Setup GUI
        # ----------------------------------------------------

        self.setup_ui()


    # ========================================================
    # SETUP USER INTERFACE
    # ========================================================

    def setup_ui(self):

        # ----------------------------------------------------
        # Main layout
        # ----------------------------------------------------

        # Scroll area to prevent label clipping
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        container = QWidget()
        main_layout = QVBoxLayout()


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
                font-size: 22px;
                font-weight: bold;
                padding: 12px;
            }
            """
        )

        main_layout.addWidget(title)


        # ====================================================
        # SIGNAL INPUT GROUP
        # ====================================================

        input_group = QGroupBox(
            "Signal Input"
        )

        input_layout = QHBoxLayout()


        # ----------------------------------------------------
        # File path label
        # ----------------------------------------------------

        self.file_path_label = QLabel(
            "No file selected"
        )

        self.file_path_label.setWordWrap(
            True
        )


        # ----------------------------------------------------
        # Browse button
        # ----------------------------------------------------

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

        sample_rate_label = QLabel(
            "Sample Rate:"
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
            sample_rate_label,
            0,
            0
        )

        iq_layout.addWidget(
            self.sample_rate_input,
            0,
            1
        )


        # ----------------------------------------------------
        # Data type
        # ----------------------------------------------------

        data_type_label = QLabel(
            "Data Type:"
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

        # Raw IQ test data generated as float32
        self.data_type_combo.setCurrentText(
            "float32"
        )


        iq_layout.addWidget(
            data_type_label,
            1,
            0
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
        # LOAD SIGNAL BUTTON
        # ====================================================

        self.load_button = QPushButton(
            "LOAD SIGNAL"
        )

        self.load_button.setMinimumHeight(
            45
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


        # ----------------------------------------------------
        # Format
        # ----------------------------------------------------

        information_layout.addWidget(
            QLabel("Format:"),
            0,
            0
        )

        self.format_value = QLabel(
            "-"
        )

        information_layout.addWidget(
            self.format_value,
            0,
            1
        )


        # ----------------------------------------------------
        # Sample rate
        # ----------------------------------------------------

        information_layout.addWidget(
            QLabel("Sample Rate:"),
            1,
            0
        )

        self.info_sample_rate = QLabel(
            "-"
        )

        information_layout.addWidget(
            self.info_sample_rate,
            1,
            1
        )


        # ----------------------------------------------------
        # Number of samples
        # ----------------------------------------------------

        information_layout.addWidget(
            QLabel("Number of Samples:"),
            2,
            0
        )

        self.info_num_samples = QLabel(
            "-"
        )

        information_layout.addWidget(
            self.info_num_samples,
            2,
            1
        )


        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        information_layout.addWidget(
            QLabel("Duration:"),
            3,
            0
        )

        self.info_duration = QLabel(
            "-"
        )

        information_layout.addWidget(
            self.info_duration,
            3,
            1
        )


        # ----------------------------------------------------
        # Data type
        # ----------------------------------------------------

        information_layout.addWidget(
            QLabel("Data Type:"),
            4,
            0
        )

        self.info_data_type = QLabel(
            "-"
        )

        information_layout.addWidget(
            self.info_data_type,
            4,
            1
        )


        # ----------------------------------------------------
        # Channels
        # ----------------------------------------------------

        information_layout.addWidget(
            QLabel("Channels:"),
            5,
            0
        )

        self.info_channels = QLabel(
            "-"
        )

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
        # STATUS
        # ====================================================

        self.status_label = QLabel(
            "Status: Ready"
        )

        self.status_label.setWordWrap(
            True
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
        # SIGNAL ANALYSIS SECTION
        # ====================================================

        main_layout.addWidget(
            self.analysis_widget
        )


        # ====================================================
        # SET MAIN LAYOUT
        # ====================================================

        container.setLayout(
            main_layout
        )

        scroll.setWidget(
            container
        )

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        self.setLayout(
            outer_layout
        )


        # ====================================================
        # WINDOW SETTINGS
        # ====================================================

        self.setWindowTitle(
            "SIH Signal Intelligence Toolkit"
        )

        self.setMinimumWidth(1050)

        self.resize(
            1100,
            800
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


        # ----------------------------------------------------
        # Check whether user selected a file
        # ----------------------------------------------------

        if not file_path:
            return


        # ----------------------------------------------------
        # Save selected file path
        # ----------------------------------------------------

        self.current_filepath = file_path


        # ----------------------------------------------------
        # Display file path
        # ----------------------------------------------------

        self.file_path_label.setText(
            file_path
        )


        # ----------------------------------------------------
        # Update status
        # ----------------------------------------------------

        self.status_label.setText(
            "Status: File selected. "
            "Click LOAD SIGNAL."
        )


    # ========================================================
    # LOAD SIGNAL
    # ========================================================

    def load_signal(self):

        # ----------------------------------------------------
        # Check file selection
        # ----------------------------------------------------

        if not self.current_filepath:

            QMessageBox.warning(
                self,
                "No File Selected",
                "Please select an IQ or WAV file first."
            )

            return


        # ----------------------------------------------------
        # Update status
        # ----------------------------------------------------

        self.status_label.setText(
            "Status: Loading signal..."
        )

        QApplication.processEvents()


        try:

            # =================================================
            # GET SAMPLE RATE FROM GUI
            # =================================================

            sample_rate = float(
                self.sample_rate_input.value()
            )


            # -------------------------------------------------
            # Check sample rate
            # -------------------------------------------------

            if sample_rate <= 0:

                raise ValueError(
                    "Sample rate must be greater than 0 Hz."
                )


            # =================================================
            # GET DATA TYPE FROM GUI
            # =================================================

            iq_dtype = (
                self.data_type_combo.currentText()
            )


            # =================================================
            # LOAD USING TEAM INGESTION MODULE
            # =================================================
            #
            # IMPORTANT:
            #
            # Raw .iq files do NOT contain sample-rate
            # information inside the file.
            #
            # Therefore we explicitly pass:
            #
            #     sample_rate
            #     iq_dtype
            #
            # to the ingestion module.
            #
            # =================================================

            result = load_signal_file(
                self.current_filepath,
                sample_rate=sample_rate,
                iq_dtype=iq_dtype
            )


            # -------------------------------------------------
            # Check ingestion result
            # -------------------------------------------------

            if result is None:

                raise ValueError(
                    "Signal loader returned no result."
                )


            # -------------------------------------------------
            # Check required keys
            # -------------------------------------------------

            if "samples" not in result:

                raise KeyError(
                    "Signal loader result does not "
                    "contain 'samples'."
                )


            if "sample_rate" not in result:

                raise KeyError(
                    "Signal loader result does not "
                    "contain 'sample_rate'."
                )


            # =================================================
            # SAVE INGESTION RESULT
            # =================================================

            self.current_result = result

            self.current_signal = (
                result["samples"]
            )


            # =================================================
            # DISPLAY SIGNAL INFORMATION
            # =================================================

            self.display_signal_information(
                self.current_result
            )


            # =================================================
            # RUN SIGNAL ANALYSIS
            # =================================================

            self.current_result = analyze_signal(
                self.current_result
            )


            # =================================================
            # CHECK ANALYSIS RESULT
            # =================================================

            if (
                self.current_result.get(
                    "analysis_status"
                )
                != "success"
            ):

                raise ValueError(
                    "Signal analysis failed: "
                    +
                    str(
                        self.current_result.get(
                            "error",
                            "Unknown error"
                        )
                    )
                )


            # =================================================
            # DISPLAY WAVEFORM / FFT / PSD
            # =================================================

            self.analysis_widget.display_analysis(
                self.current_result
            )


            # =================================================
            # SUCCESS STATUS
            # =================================================

            self.status_label.setText(
                "Status: ✓ Signal loaded and "
                "analyzed successfully."
            )


        except Exception as error:

            # -------------------------------------------------
            # Clear old signal
            # -------------------------------------------------

            self.current_signal = None

            self.current_result = None


            # -------------------------------------------------
            # Clear graphs
            # -------------------------------------------------

            self.analysis_widget.clear_plots()


            # -------------------------------------------------
            # Display error
            # -------------------------------------------------

            self.status_label.setText(
                "Status: ✗ Error - "
                + str(error)
            )


            QMessageBox.critical(
                self,
                "Signal Loading / Analysis Error",
                str(error)
            )


    # ========================================================
    # DISPLAY SIGNAL INFORMATION
    # ========================================================

    def display_signal_information(
        self,
        result
    ):

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

        sample_rate = result.get(
            "sample_rate",
            None
        )

        if sample_rate is not None:

            self.info_sample_rate.setText(
                f"{float(sample_rate):,.2f} Hz"
            )

        else:

            self.info_sample_rate.setText(
                "-"
            )


        # ----------------------------------------------------
        # Number of samples
        # ----------------------------------------------------

        samples = result.get(
            "samples",
            None
        )

        if samples is not None:

            self.info_num_samples.setText(
                f"{len(samples):,}"
            )

        else:

            # Some versions of the ingestion
            # module may already provide num_samples.

            num_samples = result.get(
                "num_samples",
                None
            )

            if num_samples is not None:

                self.info_num_samples.setText(
                    f"{num_samples:,}"
                )

            else:

                self.info_num_samples.setText(
                    "-"
                )


        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration = result.get(
            "duration_sec",
            None
        )

        if duration is not None:

            self.info_duration.setText(
                f"{float(duration):.6f} sec"
            )

        elif (
            samples is not None
            and sample_rate is not None
            and float(sample_rate) > 0
        ):

            calculated_duration = (
                len(samples)
                / float(sample_rate)
            )

            self.info_duration.setText(
                f"{calculated_duration:.6f} sec"
            )

        else:

            self.info_duration.setText(
                "-"
            )


        # ----------------------------------------------------
        # Data type
        # ----------------------------------------------------

        if samples is not None:

            self.info_data_type.setText(
                str(
                    samples.dtype
                )
            )

        else:

            self.info_data_type.setText(
                "-"
            )


        # ----------------------------------------------------
        # Channels
        # ----------------------------------------------------

        channels = result.get(
            "channels",
            None
        )

        if channels is not None:

            self.info_channels.setText(
                str(channels)
            )

        else:

            # IQ complex data normally represents
            # I and Q components.

            if (
                samples is not None
                and np_is_complex(samples)
            ):

                self.info_channels.setText(
                    "2 (I + Q)"
                )

            else:

                self.info_channels.setText(
                    "-"
                )


# ============================================================
# HELPER FUNCTION
# ============================================================

def np_is_complex(array):

    """
    Check whether a NumPy array contains complex data.
    """

    try:

        import numpy as np

        return np.iscomplexobj(
            array
        )

    except Exception:

        return False


# ============================================================
# APPLICATION ENTRY POINT
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
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()
