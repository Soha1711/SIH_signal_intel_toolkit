"""
SIH 2026
Signal Analysis Dashboard

Displays:
1. Time-Domain I/Q Waveform
2. FFT / Magnitude Spectrum
3. Power Spectral Density
4. Waterfall / Spectrogram
5. Constellation Diagram
"""

import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas
)

from matplotlib.figure import Figure
from scipy.signal import spectrogram


class SignalAnalysisWidget(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setup_ui()

    # =========================================================
    # CREATE DASHBOARD
    # =========================================================

    def setup_ui(self):

        main_layout = QVBoxLayout()

        # -----------------------------------------------------
        # TABS
        # -----------------------------------------------------

        self.tabs = QTabWidget()

        # Create five graph pages

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

        # -----------------------------------------------------
        # CREATE FIGURES
        # -----------------------------------------------------

        self.waveform_figure = Figure(
            figsize=(9, 5)
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

        # -----------------------------------------------------

        self.fft_figure = Figure(
            figsize=(9, 5)
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

        # -----------------------------------------------------

        self.psd_figure = Figure(
            figsize=(9, 5)
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

        # -----------------------------------------------------

        self.waterfall_figure = Figure(
            figsize=(9, 5)
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

        # -----------------------------------------------------

        self.constellation_figure = Figure(
            figsize=(9, 5)
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

        # -----------------------------------------------------
        # ADD TABS TO MAIN LAYOUT
        # -----------------------------------------------------

        main_layout.addWidget(
            self.tabs
        )

        self.setLayout(
            main_layout
        )

    # =========================================================
    # DISPLAY ALL ANALYSIS
    # =========================================================

    def display_analysis(self, result):

        if result.get(
            "analysis_status"
        ) != "success":

            return

        self.plot_waveform(result)

        self.plot_fft(result)

        self.plot_psd(result)

        self.plot_waterfall(result)

        self.plot_constellation(result)

    # =========================================================
    # 1. WAVEFORM
    # =========================================================

    def plot_waveform(self, result):

        self.waveform_figure.clear()

        ax = self.waveform_figure.add_subplot(111)

        time = result.get(
            "waveform_time"
        )

        i_data = result.get(
            "waveform_i"
        )

        q_data = result.get(
            "waveform_q"
        )

        if time is None:

            signal = np.asarray(
                result["samples"]
            )

            sample_rate = float(
                result["sample_rate"]
            )

            max_points = min(
                len(signal),
                5000
            )

            signal = signal[:max_points]

            time = (
                np.arange(
                    len(signal)
                )
                / sample_rate
            )

            i_data = np.real(signal)

            q_data = np.imag(signal)

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

        ax.grid(True)

        ax.legend()

        self.waveform_figure.tight_layout()

        self.waveform_canvas.draw()

    # =========================================================
    # 2. FFT
    # =========================================================

    def plot_fft(self, result):

        self.fft_figure.clear()

        ax = self.fft_figure.add_subplot(111)

        frequencies = result.get(
            "fft_frequency"
        )

        magnitude = result.get(
            "fft_magnitude_db"
        )

        if frequencies is None:

            return

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

        ax.grid(True)

        self.fft_figure.tight_layout()

        self.fft_canvas.draw()

    # =========================================================
    # 3. PSD
    # =========================================================

    def plot_psd(self, result):

        self.psd_figure.clear()

        ax = self.psd_figure.add_subplot(111)

        frequencies = result.get(
            "psd_frequency"
        )

        psd = result.get(
            "psd_db"
        )

        if frequencies is None:

            return

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

        ax.grid(True)

        self.psd_figure.tight_layout()

        self.psd_canvas.draw()

    # =========================================================
    # 4. WATERFALL / SPECTROGRAM
    # =========================================================

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

        # Spectrogram

        frequencies, times, power = spectrogram(
            signal,
            fs=sample_rate,
            nperseg=min(
                1024,
                len(signal)
            ),
            noverlap=min(
                768,
                len(signal) // 2
            ),
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

    # =========================================================
    # 5. CONSTELLATION
    # =========================================================

    def plot_constellation(self, result):

        self.constellation_figure.clear()

        ax = self.constellation_figure.add_subplot(111)

        signal = np.asarray(
            result["samples"]
        )

        if len(signal) == 0:

            return

        # Limit number of points

        max_points = 10000

        if len(signal) > max_points:

            indices = np.linspace(
                0,
                len(signal) - 1,
                max_points
            ).astype(int)

            signal = signal[
                indices
            ]

        i_data = np.real(
            signal
        )

        q_data = np.imag(
            signal
        )

        ax.scatter(
            i_data,
            q_data,
            s=4,
            alpha=0.5
        )

        ax.axhline(
            0,
            linewidth=0.8
        )

        ax.axvline(
            0,
            linewidth=0.8
        )

        ax.set_title(
            "I/Q Constellation Diagram"
        )

        ax.set_xlabel(
            "In-Phase (I)"
        )

        ax.set_ylabel(
            "Quadrature (Q)"
        )

        ax.grid(True)

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        self.constellation_figure.tight_layout()

        self.constellation_canvas.draw()

    # =========================================================
    # CLEAR PLOTS
    # =========================================================

    def clear_plots(self):

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
