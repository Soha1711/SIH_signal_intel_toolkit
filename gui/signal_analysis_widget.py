"""
SIH 2026
Signal Intelligence Toolkit
Signal Analysis GUI Widget

Displays:
1. Time-Domain I/Q Waveform
2. FFT / Magnitude Spectrum
3. Power Spectral Density
4. Waterfall / Spectrogram
5. I/Q Constellation

Also provides:
- Export CSV
- Export JSON
- Save Plot
"""

import json
import numpy as np

from scipy.signal import spectrogram

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QPushButton,
    QFileDialog,
    QMessageBox,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas
)

from matplotlib.figure import Figure


class SignalAnalysisWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_result = None

        self.setup_ui()

    # ============================================================
    # CREATE GUI
    # ============================================================

    def setup_ui(self):

        main_layout = QVBoxLayout()

        # ========================================================
        # TABS
        # ========================================================

        self.tabs = QTabWidget()

        # Pages
        self.waveform_page = QWidget()
        self.fft_page = QWidget()
        self.psd_page = QWidget()
        self.waterfall_page = QWidget()
        self.constellation_page = QWidget()

        # Add tabs
        self.tabs.addTab(
            self.waveform_page,
            "Waveform"
        )

        self.tabs.addTab(
            self.fft_page,
            "FFT / Spectrum"
        )

        self.tabs.addTab(
            self.psd_page,
            "PSD"
        )

        self.tabs.addTab(
            self.waterfall_page,
            "Waterfall"
        )

        self.tabs.addTab(
            self.constellation_page,
            "Constellation"
        )

        # ========================================================
        # WAVEFORM FIGURE
        # ========================================================

        self.waveform_figure = Figure(
            figsize=(10, 5)
        )

        self.waveform_canvas = FigureCanvas(
            self.waveform_figure
        )

        waveform_layout = QVBoxLayout(
            self.waveform_page
        )

        waveform_layout.addWidget(
            self.waveform_canvas
        )

        # ========================================================
        # FFT FIGURE
        # ========================================================

        self.fft_figure = Figure(
            figsize=(10, 5)
        )

        self.fft_canvas = FigureCanvas(
            self.fft_figure
        )

        fft_layout = QVBoxLayout(
            self.fft_page
        )

        fft_layout.addWidget(
            self.fft_canvas
        )

        # ========================================================
        # PSD FIGURE
        # ========================================================

        self.psd_figure = Figure(
            figsize=(10, 5)
        )

        self.psd_canvas = FigureCanvas(
            self.psd_figure
        )

        psd_layout = QVBoxLayout(
            self.psd_page
        )

        psd_layout.addWidget(
            self.psd_canvas
        )

        # ========================================================
        # WATERFALL FIGURE
        # ========================================================

        self.waterfall_figure = Figure(
            figsize=(10, 5)
        )

        self.waterfall_canvas = FigureCanvas(
            self.waterfall_figure
        )

        waterfall_layout = QVBoxLayout(
            self.waterfall_page
        )

        waterfall_layout.addWidget(
            self.waterfall_canvas
        )

        # ========================================================
        # CONSTELLATION FIGURE
        # ========================================================

        self.constellation_figure = Figure(
            figsize=(10, 7)
        )

        self.constellation_canvas = FigureCanvas(
            self.constellation_figure
        )

        constellation_layout = QVBoxLayout(
            self.constellation_page
        )

        constellation_layout.addWidget(
            self.constellation_canvas
        )

        # ========================================================
        # EXPORT BUTTONS
        # ========================================================

        export_layout = QHBoxLayout()

        self.export_csv_button = QPushButton(
            "Export CSV"
        )

        self.export_json_button = QPushButton(
            "Export JSON"
        )

        self.save_plot_button = QPushButton(
            "Save Plot"
        )

        self.export_csv_button.clicked.connect(
            self.export_csv
        )

        self.export_json_button.clicked.connect(
            self.export_json
        )

        self.save_plot_button.clicked.connect(
            self.save_plot
        )

        export_layout.addWidget(
            self.export_csv_button
        )

        export_layout.addWidget(
            self.export_json_button
        )

        export_layout.addWidget(
            self.save_plot_button
        )

        # ========================================================
        # ADD EVERYTHING TO MAIN LAYOUT
        # ========================================================

        main_layout.addWidget(
            self.tabs
        )

        main_layout.addLayout(
            export_layout
        )

        self.setLayout(
            main_layout
        )

    # ============================================================
    # DISPLAY ALL ANALYSIS
    # ============================================================

    def display_analysis(self, result):

        if result.get(
            "analysis_status"
        ) != "success":

            return

        self.current_result = result

        self.plot_waveform(
            result
        )

        self.plot_fft(
            result
        )

        self.plot_psd(
            result
        )

        self.plot_waterfall(
            result
        )

        self.plot_constellation(
            result
        )

    # ============================================================
    # 1. WAVEFORM
    # ============================================================

    def plot_waveform(self, result):

        self.waveform_figure.clear()

        ax = self.waveform_figure.add_subplot(111)

        signal = np.asarray(
            result["samples"]
        )

        sample_rate = float(
            result["sample_rate"]
        )

        # Limit points for GUI
        max_points = 5000

        if len(signal) > max_points:

            signal = signal[:max_points]

        time = (
            np.arange(
                len(signal)
            )
            / sample_rate
        )

        i_data = np.real(
            signal
        )

        q_data = np.imag(
            signal
        )

        ax.plot(
            time,
            i_data,
            label="I"
        )

        ax.plot(
            time,
            q_data,
            label="Q"
        )

        ax.set_title(
            "Time-Domain I/Q Waveform"
        )

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.legend()

        self.waveform_figure.tight_layout()

        self.waveform_canvas.draw()

    # ============================================================
    # 2. FFT / MAGNITUDE SPECTRUM
    # ============================================================

    def plot_fft(self, result):

        self.fft_figure.clear()

        ax = self.fft_figure.add_subplot(111)

        frequencies = result.get(
            "fft_frequency"
        )

        magnitude = result.get(
            "fft_magnitude_db"
        )

        if frequencies is None or magnitude is None:

            signal = np.asarray(
                result["samples"]
            )

            sample_rate = float(
                result["sample_rate"]
            )

            max_samples = min(
                len(signal),
                65536
            )

            signal = signal[
                :max_samples
            ]

            fft = np.fft.fftshift(
                np.fft.fft(signal)
            )

            frequencies = np.fft.fftshift(
                np.fft.fftfreq(
                    len(signal),
                    d=1 / sample_rate
                )
            )

            magnitude = (
                20
                * np.log10(
                    np.abs(fft)
                    / max(
                        np.max(
                            np.abs(fft)
                        ),
                        1e-12
                    )
                    + 1e-12
                )
            )

        ax.plot(
            frequencies,
            magnitude
        )

        ax.set_title(
            "FFT / Magnitude Spectrum"
        )

        ax.set_xlabel(
            "Frequency (Hz)"
        )

        ax.set_ylabel(
            "Magnitude (dB)"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        self.fft_figure.tight_layout()

        self.fft_canvas.draw()

    # ============================================================
    # 3. PSD
    # ============================================================

    def plot_psd(self, result):

        self.psd_figure.clear()

        ax = self.psd_figure.add_subplot(111)

        frequencies = result.get(
            "psd_frequency"
        )

        psd = result.get(
            "psd_db"
        )

        if frequencies is None or psd is None:

            signal = np.asarray(
                result["samples"]
            )

            sample_rate = float(
                result["sample_rate"]
            )

            frequencies, _, power = spectrogram(
                signal,
                fs=sample_rate,
                nperseg=min(
                    1024,
                    len(signal)
                ),
                noverlap=512
                if len(signal) > 1024
                else 0,
                return_onesided=False,
                mode="psd"
            )

            psd = np.mean(
                power,
                axis=1
            )

            frequencies = np.fft.fftshift(
                frequencies
            )

            psd = np.fft.fftshift(
                psd
            )

            psd = 10 * np.log10(
                psd + 1e-12
            )

        ax.plot(
            frequencies,
            psd
        )

        ax.set_title(
            "Power Spectral Density"
        )

        ax.set_xlabel(
            "Frequency (Hz)"
        )

        ax.set_ylabel(
            "Power (dB/Hz)"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        self.psd_figure.tight_layout()

        self.psd_canvas.draw()

    # ============================================================
    # 4. WATERFALL / SPECTROGRAM
    # ============================================================

    def plot_waterfall(self, result):

        self.waterfall_figure.clear()

        ax = self.waterfall_figure.add_subplot(111)

        signal = np.asarray(
            result["samples"]
        )

        sample_rate = float(
            result["sample_rate"]
        )

        # Limit data for GUI performance
        max_samples = 200000

        if len(signal) > max_samples:

            signal = signal[
                :max_samples
            ]

        if len(signal) < 32:

            ax.set_title(
                "Waterfall / Spectrogram"
            )

            ax.text(
                0.5,
                0.5,
                "Not enough samples",
                ha="center",
                va="center"
            )

            self.waterfall_canvas.draw()

            return

        # ========================================================
        # SPECTROGRAM
        # ========================================================

        nperseg = min(
            1024,
            len(signal)
        )

        noverlap = min(
            768,
            nperseg - 1
        )

        frequencies, times, power = spectrogram(
            signal,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False,
            mode="magnitude"
        )

        # Shift zero frequency to center
        frequencies = np.fft.fftshift(
            frequencies
        )

        power = np.fft.fftshift(
            power,
            axes=0
        )

        power_db = (
            20
            * np.log10(
                power + 1e-12
            )
        )

        mesh = ax.pcolormesh(
            times,
            frequencies,
            power_db,
            shading="auto"
        )

        self.waterfall_figure.colorbar(
            mesh,
            ax=ax,
            label="Magnitude (dB)"
        )

        ax.set_title(
            "Waterfall / Spectrogram"
        )

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "Frequency (Hz)"
        )

        self.waterfall_figure.tight_layout()

        self.waterfall_canvas.draw()

    # ============================================================
    # 5. CONSTELLATION
    # ============================================================

    def plot_constellation(self, result):

        self.constellation_figure.clear()

        ax = self.constellation_figure.add_subplot(
            111
        )

        signal = np.asarray(
            result["samples"]
        )

        if len(signal) == 0:

            ax.set_title(
                "I/Q Constellation Diagram"
            )

            self.constellation_canvas.draw()

            return

        # ========================================================
        # LIMIT POINTS
        # ========================================================

        max_points = 10000

        if len(signal) > max_points:

            step = max(
                1,
                len(signal) // max_points
            )

            signal = signal[
                ::step
            ]

        i_data = np.real(
            signal
        )

        q_data = np.imag(
            signal
        )

        # ========================================================
        # NORMALIZE
        # ========================================================

        amplitude = np.max(
            np.abs(signal)
        )

        if amplitude > 0:

            i_data = (
                i_data / amplitude
            )

            q_data = (
                q_data / amplitude
            )

        # ========================================================
        # PLOT
        # ========================================================

        ax.scatter(
            i_data,
            q_data,
            s=8,
            alpha=0.6
        )

        # Reference axes
        ax.axhline(
            0,
            linewidth=0.8
        )

        ax.axvline(
            0,
            linewidth=0.8
        )

        # ========================================================
        # MODULATION INFORMATION
        # ========================================================

        detected_type = result.get(
            "detected_modulation",
            result.get(
                "modulation",
                "Unknown"
            )
        )

        confidence = result.get(
            "modulation_confidence",
            result.get(
                "confidence",
                0.0
            )
        )

        if isinstance(
            confidence,
            (int, float)
        ):

            if confidence <= 1:

                confidence_text = (
                    f"{confidence:.1%}"
                )

            else:

                confidence_text = (
                    f"{confidence:.1f}%"
                )

        else:

            confidence_text = str(
                confidence
            )

        # ========================================================
        # TITLE
        # ========================================================

        title = (
            "I/Q Constellation Diagram"
        )

        if detected_type != "Unknown":

            title += (
                f"\nDetected: {detected_type}"
                f"  |  Confidence: "
                f"{confidence_text}"
            )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            "In-Phase (I)"
        )

        ax.set_ylabel(
            "Quadrature (Q)"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        self.constellation_figure.tight_layout()

        self.constellation_canvas.draw()

    # ============================================================
    # EXPORT CSV
    # ============================================================

    def export_csv(self):

        if self.current_result is None:

            QMessageBox.warning(
                self,
                "No Signal",
                "Please load and analyze a signal first."
            )

            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Signal Data",
            "signal_analysis.csv",
            "CSV Files (*.csv)"
        )

        if not file_path:

            return

        try:

            result = self.current_result

            signal = np.asarray(
                result["samples"]
            )

            sample_rate = float(
                result["sample_rate"]
            )

            max_points = min(
                len(signal),
                100000
            )

            signal = signal[
                :max_points
            ]

            time = (
                np.arange(
                    len(signal)
                )
                / sample_rate
            )

            data = np.column_stack(
                (
                    time,
                    np.real(signal),
                    np.imag(signal),
                    np.abs(signal),
                    np.angle(signal)
                )
            )

            header = (
                "Time(s),I,Q,Magnitude,Phase(rad)"
            )

            np.savetxt(
                file_path,
                data,
                delimiter=",",
                header=header,
                comments=""
            )

            QMessageBox.information(
                self,
                "Export Successful",
                "CSV file exported successfully."
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Export Error",
                str(error)
            )

    # ============================================================
    # EXPORT JSON
    # ============================================================

    def export_json(self):

        if self.current_result is None:

            QMessageBox.warning(
                self,
                "No Signal",
                "Please load and analyze a signal first."
            )

            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis Results",
            "signal_analysis.json",
            "JSON Files (*.json)"
        )

        if not file_path:

            return

        try:

            result = self.current_result

            export_data = {}

            for key, value in result.items():

                if isinstance(
                    value,
                    np.ndarray
                ):

                    # Avoid exporting extremely
                    # large arrays unnecessarily
                    if value.size <= 100000:

                        export_data[key] = (
                            value.tolist()
                        )

                elif isinstance(
                    value,
                    np.generic
                ):

                    export_data[key] = (
                        value.item()
                    )

                elif isinstance(
                    value,
                    (str, int, float, bool)
                ) or value is None:

                    export_data[key] = value

            with open(
                file_path,
                "w",
                encoding="utf-8"
            ) as json_file:

                json.dump(
                    export_data,
                    json_file,
                    indent=4
                )

            QMessageBox.information(
                self,
                "Export Successful",
                "JSON file exported successfully."
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Export Error",
                str(error)
            )

    # ============================================================
    # SAVE CURRENT PLOT
    # ============================================================

    def save_plot(self):

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "signal_plot.png",
            "PNG Image (*.png);;"
            "JPEG Image (*.jpg);;"
            "PDF File (*.pdf)"
        )

        if not file_path:

            return

        try:

            current_index = (
                self.tabs.currentIndex()
            )

            figures = [
                self.waveform_figure,
                self.fft_figure,
                self.psd_figure,
                self.waterfall_figure,
                self.constellation_figure
            ]

            figure = figures[
                current_index
            ]

            figure.savefig(
                file_path,
                dpi=200,
                bbox_inches="tight"
            )

            QMessageBox.information(
                self,
                "Plot Saved",
                "Current plot saved successfully."
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Save Error",
                str(error)
            )

    # ============================================================
    # CLEAR PLOTS
    # ============================================================

    def clear_plots(self):

        self.current_result = None

        self.waveform_figure.clear()

        self.fft_figure.clear()

        self.psd_figure.clear()

        self.waterfall_figure.clear()

        self.constellation_figure.clear()

        self.waveform_canvas.draw()

        self.fft_canvas.draw()

        self.psd_canvas.draw()

        self.waterfall_canvas.draw()

        self.constellation_canvas.draw()
