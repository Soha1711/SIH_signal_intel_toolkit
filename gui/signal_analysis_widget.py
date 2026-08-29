"""
SIH 2026
Signal Analysis GUI Widget

Displays:
1. Time-Domain I/Q Waveform
2. FFT / Magnitude Spectrum
3. Power Spectral Density
"""

import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QScrollArea,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas
)

from matplotlib.figure import Figure


class SignalAnalysisWidget(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setup_ui()

    # ========================================================
    # CREATE GUI
    # ========================================================

    def setup_ui(self):

        outer_layout = QVBoxLayout()

        # ----------------------------------------------------
        # SIGNAL ANALYSIS GROUP
        # ----------------------------------------------------

        analysis_group = QGroupBox(
            "Signal Analysis"
        )

        analysis_layout = QVBoxLayout()

        # ----------------------------------------------------
        # SCROLL AREA
        # ----------------------------------------------------

        scroll_area = QScrollArea()

        scroll_area.setWidgetResizable(
            True
        )

        # ----------------------------------------------------
        # WIDGET INSIDE SCROLL AREA
        # ----------------------------------------------------

        graph_widget = QWidget()

        graph_layout = QVBoxLayout()

        # ====================================================
        # WAVEFORM FIGURE
        # ====================================================

        self.waveform_figure = Figure(
            figsize=(9, 4)
        )

        self.waveform_canvas = FigureCanvas(
            self.waveform_figure
        )

        graph_layout.addWidget(
            self.waveform_canvas
        )

        # ====================================================
        # FFT FIGURE
        # ====================================================

        self.fft_figure = Figure(
            figsize=(9, 4)
        )

        self.fft_canvas = FigureCanvas(
            self.fft_figure
        )

        graph_layout.addWidget(
            self.fft_canvas
        )

        # ====================================================
        # PSD FIGURE
        # ====================================================

        self.psd_figure = Figure(
            figsize=(9, 4)
        )

        self.psd_canvas = FigureCanvas(
            self.psd_figure
        )

        graph_layout.addWidget(
            self.psd_canvas
        )

        # ----------------------------------------------------
        # SET GRAPH WIDGET
        # ----------------------------------------------------

        graph_widget.setLayout(
            graph_layout
        )

        scroll_area.setWidget(
            graph_widget
        )

        # ----------------------------------------------------
        # ADD SCROLL AREA TO ANALYSIS GROUP
        # ----------------------------------------------------

        analysis_layout.addWidget(
            scroll_area
        )

        analysis_group.setLayout(
            analysis_layout
        )

        outer_layout.addWidget(
            analysis_group
        )

        self.setLayout(
            outer_layout
        )

    # ========================================================
    # DISPLAY ALL ANALYSIS
    # ========================================================

    def display_analysis(self, result):

        if result.get(
            "analysis_status"
        ) != "success":

            return

        self.plot_waveform(
            result
        )

        self.plot_fft(
            result
        )

        self.plot_psd(
            result
        )

    # ========================================================
    # 1. WAVEFORM
    # ========================================================

    def plot_waveform(self, result):

        self.waveform_figure.clear()

        ax = self.waveform_figure.add_subplot(
            111
        )

        # Get waveform data
        time = result[
            "waveform_time"
        ]

        i_data = result[
            "waveform_i"
        ]

        q_data = result[
            "waveform_q"
        ]

        # Convert seconds to milliseconds
        time_ms = (
            time * 1000
        )

        # Plot I
        ax.plot(
            time_ms,
            i_data,
            label="I (In-phase)",
            linewidth=1
        )

        # Plot Q
        ax.plot(
            time_ms,
            q_data,
            label="Q (Quadrature)",
            linewidth=1
        )

        # Title
        ax.set_title(
            "Time-Domain I/Q Waveform",
            fontsize=14,
            fontweight="bold"
        )

        # Labels
        ax.set_xlabel(
            "Time (ms)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        # Grid
        ax.grid(
            True,
            alpha=0.3
        )

        # Legend
        ax.legend()

        # Improve layout
        self.waveform_figure.tight_layout()

        # Refresh graph
        self.waveform_canvas.draw()

    # ========================================================
    # 2. FFT
    # ========================================================

    def plot_fft(self, result):

        self.fft_figure.clear()

        ax = self.fft_figure.add_subplot(
            111
        )

        # Get FFT data
        frequency = result[
            "fft_frequency"
        ]

        magnitude_db = result[
            "fft_magnitude_db"
        ]

        # Convert Hz to kHz
        frequency_khz = (
            frequency / 1000
        )

        # Plot FFT
        ax.plot(
            frequency_khz,
            magnitude_db,
            linewidth=1
        )

        # Title
        ax.set_title(
            "FFT / Magnitude Spectrum",
            fontsize=14,
            fontweight="bold"
        )

        # Labels
        ax.set_xlabel(
            "Frequency (kHz)"
        )

        ax.set_ylabel(
            "Magnitude (dB)"
        )

        # Grid
        ax.grid(
            True,
            alpha=0.3
        )

        # ----------------------------------------------------
        # MARK PEAK FREQUENCY
        # ----------------------------------------------------

        peak_frequency = result[
            "peak_frequency"
        ]

        peak_index = np.argmax(
            magnitude_db
        )

        peak_magnitude = (
            magnitude_db[
                peak_index
            ]
        )

        ax.scatter(
            peak_frequency / 1000,
            peak_magnitude,
            zorder=5
        )

        ax.annotate(
            f"Peak: "
            f"{peak_frequency / 1000:.2f} kHz",
            (
                peak_frequency / 1000,
                peak_magnitude
            ),
            xytext=(10, 10),
            textcoords="offset points"
        )

        # Improve layout
        self.fft_figure.tight_layout()

        # Refresh
        self.fft_canvas.draw()

    # ========================================================
    # 3. PSD
    # ========================================================

    def plot_psd(self, result):

        self.psd_figure.clear()

        ax = self.psd_figure.add_subplot(
            111
        )

        # Get PSD data
        frequency = result[
            "psd_frequency"
        ]

        psd_db = result[
            "psd_db"
        ]

        # Convert Hz to kHz
        frequency_khz = (
            frequency / 1000
        )

        # Plot PSD
        ax.plot(
            frequency_khz,
            psd_db,
            linewidth=1
        )

        # Title
        ax.set_title(
            "Power Spectral Density",
            fontsize=14,
            fontweight="bold"
        )

        # Labels
        ax.set_xlabel(
            "Frequency (kHz)"
        )

        ax.set_ylabel(
            "PSD (dB/Hz)"
        )

        # Grid
        ax.grid(
            True,
            alpha=0.3
        )

        # Improve layout
        self.psd_figure.tight_layout()

        # Refresh
        self.psd_canvas.draw()

    # ========================================================
    # CLEAR GRAPHS
    # ========================================================

    def clear_plots(self):

        self.waveform_figure.clear()

        self.fft_figure.clear()

        self.psd_figure.clear()

        self.waveform_canvas.draw()

        self.fft_canvas.draw()

        self.psd_canvas.draw()