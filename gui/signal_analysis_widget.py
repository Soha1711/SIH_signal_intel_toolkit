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
        # ADD EVERYTHING TO MAIN LAYOUT
        # ========================================================

        main_layout.addWidget(
            self.tabs
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

        from param_estimation.modulation_classifier import (
            get_ideal_constellation,
            MODULATION_TYPES,
            MODULATION_INFO,
        )

        self.constellation_figure.clear()

        signal = np.asarray(result["samples"])

        if len(signal) == 0:
            ax = self.constellation_figure.add_subplot(111)
            ax.set_title("I/Q Constellation Diagram")
            self.constellation_canvas.draw()
            return

        # --------------------------------------------------------
        # LIMIT POINTS
        # --------------------------------------------------------

        max_points = 10000

        if len(signal) > max_points:
            step = max(1, len(signal) // max_points)
            signal = signal[::step]

        i_data = np.real(signal)
        q_data = np.imag(signal)

        # Normalize
        amplitude = np.max(np.abs(signal))
        if amplitude > 0:
            i_data = i_data / amplitude
            q_data = q_data / amplitude

        # --------------------------------------------------------
        # MODULATION INFO
        # --------------------------------------------------------

        detected_type = result.get(
            "detected_modulation",
            result.get("modulation", "Unknown")
        )

        confidence = result.get(
            "modulation_confidence",
            result.get("confidence", 0.0)
        )

        if isinstance(confidence, (int, float)):
            if confidence <= 1:
                confidence_pct = int(round(confidence * 100))
            else:
                confidence_pct = int(round(confidence))
        else:
            confidence_pct = 0

        # --------------------------------------------------------
        # LAYOUT: main plot on top, 5 reference thumbnails below
        # --------------------------------------------------------

        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(
            2, 5,
            figure=self.constellation_figure,
            height_ratios=[3, 1],
            hspace=0.45,
            wspace=0.40,
        )

        # ========================================================
        # MAIN CONSTELLATION DIAGRAM (top, spans all 5 columns)
        # ========================================================

        ax_main = self.constellation_figure.add_subplot(gs[0, :])

        # Signal scatter
        ax_main.scatter(
            i_data, q_data,
            s=10, alpha=0.5, color="#4fc3f7",
            label="Signal", zorder=2,
        )

        # Overlay ideal constellation markers
        ideal_points = get_ideal_constellation(detected_type)
        if ideal_points is not None:
            ax_main.scatter(
                np.real(ideal_points),
                np.imag(ideal_points),
                s=200, marker="X", color="#e53935",
                linewidths=1.5, edgecolors="#b71c1c",
                label=f"Ideal {detected_type}", zorder=3,
            )

        # Reference axes
        ax_main.axhline(0, linewidth=0.6, color="#888888", zorder=1)
        ax_main.axvline(0, linewidth=0.6, color="#888888", zorder=1)

        # Title
        title_str = "5. Constellation Diagram"
        if detected_type != "Unknown":
            title_str += (
                f"\nDetected: {detected_type}"
                f" (Confidence: {confidence_pct}%)"
            )

        ax_main.set_title(title_str, fontsize=13, fontweight="bold")
        ax_main.set_xlabel("I (In-phase)")
        ax_main.set_ylabel("Q (Quadrature)")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_aspect("equal", adjustable="box")
        ax_main.legend(loc="upper right", fontsize=8, framealpha=0.7)

        # ========================================================
        # REFERENCE CONSTELLATION THUMBNAILS (bottom row)
        # ========================================================

        ref_types = ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM"]
        ref_colors = ["#42a5f5", "#ab47bc", "#ff9800", "#e91e63", "#26a69a"]

        for col, mod_name in enumerate(ref_types):
            ax_ref = self.constellation_figure.add_subplot(gs[1, col])

            pts = get_ideal_constellation(mod_name)
            if pts is not None:
                ax_ref.scatter(
                    np.real(pts), np.imag(pts),
                    s=28, color=ref_colors[col], zorder=2,
                )

            info = MODULATION_INFO.get(mod_name, {})
            bps = info.get("bits_per_symbol", "?")

            ax_ref.set_title(
                f"{mod_name}\n{bps} bits/symbol",
                fontsize=8, fontweight="bold",
            )

            ax_ref.set_xlim(-1.3, 1.3)
            ax_ref.set_ylim(-1.3, 1.3)
            ax_ref.set_aspect("equal", adjustable="box")
            ax_ref.tick_params(labelsize=6)

            # Highlight the detected modulation with a green border
            if mod_name == detected_type:
                for spine in ax_ref.spines.values():
                    spine.set_edgecolor("#4caf50")
                    spine.set_linewidth(3)
            else:
                for spine in ax_ref.spines.values():
                    spine.set_linewidth(0.5)

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
