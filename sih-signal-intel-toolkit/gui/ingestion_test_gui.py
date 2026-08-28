import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QComboBox,
    QMessageBox,
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.iq_wav_reader import load_signal_file


class IngestionTestGUI(QWidget):

    def __init__(self):
        super().__init__()

        self.current_signal = None
        self.current_result = None

        self.setWindowTitle(
            "SIH Signal Intelligence Toolkit"
        )

        self.resize(900, 650)

        self.setup_ui()


    # ======================================================
    # UI SETUP
    # ======================================================

    def setup_ui(self):

        main_layout = QVBoxLayout()

        # --------------------------------------------------
        # TITLE
        # --------------------------------------------------

        title = QLabel(
            "SIH SIGNAL INTELLIGENCE TOOLKIT"
        )

        title.setStyleSheet(
            """
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 15px;
            }
            """
        )

        main_layout.addWidget(title)


        # --------------------------------------------------
        # FILE SELECTION
        # --------------------------------------------------

        file_group = QGroupBox(
            "Signal Input"
        )

        file_layout = QHBoxLayout()

        self.file_path = QLineEdit()

        self.file_path.setPlaceholderText(
            "Select a .IQ or .WAV file..."
        )

        self.browse_button = QPushButton(
            "Browse..."
        )

        self.browse_button.clicked.connect(
            self.browse_file
        )

        file_layout.addWidget(
            self.file_path
        )

        file_layout.addWidget(
            self.browse_button
        )

        file_group.setLayout(
            file_layout
        )

        main_layout.addWidget(
            file_group
        )


        # --------------------------------------------------
        # IQ PARAMETERS
        # --------------------------------------------------

        self.iq_group = QGroupBox(
            "IQ File Parameters"
        )

        iq_layout = QFormLayout()

        self.sample_rate_input = QSpinBox()

        self.sample_rate_input.setRange(
            1,
            1_000_000_000
        )

        self.sample_rate_input.setValue(
            2_000_000
        )

        self.sample_rate_input.setSuffix(
            " Hz"
        )


        self.dtype_input = QComboBox()

        self.dtype_input.addItems(
            [
                "float32",
                "int16",
                "int8",
            ]
        )


        iq_layout.addRow(
            "Sample Rate:",
            self.sample_rate_input
        )

        iq_layout.addRow(
            "Data Type:",
            self.dtype_input
        )

        self.iq_group.setLayout(
            iq_layout
        )

        self.iq_group.hide()

        main_layout.addWidget(
            self.iq_group
        )


        # --------------------------------------------------
        # LOAD BUTTON
        # --------------------------------------------------

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


        # --------------------------------------------------
        # SIGNAL INFORMATION
        # --------------------------------------------------

        info_group = QGroupBox(
            "Signal Information"
        )

        info_layout = QFormLayout()

        self.format_value = QLabel("-")
        self.sample_rate_value = QLabel("-")
        self.samples_value = QLabel("-")
        self.duration_value = QLabel("-")
        self.dtype_value = QLabel("-")
        self.channels_value = QLabel("-")

        info_layout.addRow(
            "Format:",
            self.format_value
        )

        info_layout.addRow(
            "Sample Rate:",
            self.sample_rate_value
        )

        info_layout.addRow(
            "Number of Samples:",
            self.samples_value
        )

        info_layout.addRow(
            "Duration:",
            self.duration_value
        )

        info_layout.addRow(
            "Data Type:",
            self.dtype_value
        )

        info_layout.addRow(
            "Channels:",
            self.channels_value
        )

        info_group.setLayout(
            info_layout
        )

        main_layout.addWidget(
            info_group
        )


        # --------------------------------------------------
        # STATUS
        # --------------------------------------------------

        self.status_label = QLabel(
            "Status: Waiting for file..."
        )

        self.status_label.setStyleSheet(
            """
            QLabel {
                padding: 12px;
                font-weight: bold;
            }
            """
        )

        main_layout.addWidget(
            self.status_label
        )


        self.setLayout(
            main_layout
        )


    # ======================================================
    # FILE BROWSER
    # ======================================================

    def browse_file(self):

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Signal File",
            "",
            "Signal Files (*.iq *.wav);;IQ Files (*.iq);;WAV Files (*.wav)"
        )

        if not filepath:
            return

        self.file_path.setText(
            filepath
        )

        extension = Path(
            filepath
        ).suffix.lower()

        if extension == ".iq":

            self.iq_group.show()

            self.status_label.setText(
                "Status: IQ file selected — enter IQ parameters."
            )

        elif extension == ".wav":

            self.iq_group.hide()

            self.status_label.setText(
                "Status: WAV file selected."
            )


    # ======================================================
    # LOAD SIGNAL
    # ======================================================

    def load_signal(self):

        filepath = self.file_path.text().strip()

        if not filepath:

            QMessageBox.warning(
                self,
                "No File",
                "Please select an IQ or WAV file."
            )

            return


        extension = Path(
            filepath
        ).suffix.lower()


        try:

            # ----------------------------------------------
            # IQ
            # ----------------------------------------------

            if extension == ".iq":

                sample_rate = (
                    self.sample_rate_input.value()
                )

                dtype = (
                    self.dtype_input.currentText()
                )

                result = load_signal_file(
                    filepath,
                    sample_rate=sample_rate,
                    iq_dtype=dtype
                )


            # ----------------------------------------------
            # WAV
            # ----------------------------------------------

            elif extension == ".wav":

                result = load_signal_file(
                    filepath
                )


            # ----------------------------------------------
            # UNSUPPORTED
            # ----------------------------------------------

            else:

                raise ValueError(
                    "Unsupported file type."
                )


            # ----------------------------------------------
            # SAVE RESULT
            # ----------------------------------------------

            self.current_result = result

            self.current_signal = (
                result["samples"]
            )


            # ----------------------------------------------
            # DISPLAY METADATA
            # ----------------------------------------------

            self.format_value.setText(
                str(
                    result["source_format"]
                ).upper()
            )

            self.sample_rate_value.setText(
                f'{result["sample_rate"]:,.1f} Hz'
            )

            self.samples_value.setText(
                f'{result["num_samples"]:,}'
            )

            self.duration_value.setText(
                f'{result["duration_sec"]:.6f} sec'
            )

            self.dtype_value.setText(
                str(
                    result["samples"].dtype
                )
            )

            self.channels_value.setText(
                str(
                    result.get(
                        "channels",
                        "N/A"
                    )
                )
            )


            self.status_label.setText(
                "Status: ✓ Signal loaded successfully."
            )


        except Exception as error:

            self.status_label.setText(
                "Status: ✗ Failed to load signal."
            )

            QMessageBox.critical(
                self,
                "Signal Loading Error",
                str(error)
            )


# ==========================================================
# APPLICATION ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = IngestionTestGUI()

    window.show()

    sys.exit(
        app.exec()
    )