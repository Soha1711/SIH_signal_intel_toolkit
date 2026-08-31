"""
SIH 2026
Signal Analysis GUI Widget

Displays:
1. Time-Domain I/Q Waveform
2. FFT / Magnitude Spectrum
3. Power Spectral Density
4. Waterfall / Spectrogram
5. Constellation Diagram
"""

import numpy as np

from scipy.signal import spectrogram

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QSizePolicy,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas
)

from matplotlib.figure import Figure


class SignalAnalysisWidget(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setup_ui()

    # ============================================================
    # CREATE GUI
    # ============================================================

    def setup_ui(self):

        outer_layout = QVBoxLayout()

        # --------------------------------------------------------
        # SIGNAL ANALYSIS GROUP
        # --------------------------------------------------------

        analysis_group = QGroupBox(
            "Signal Analysis"
        )

        analysis_layout = QVBoxLayout()

        analysis_layout.setSpacing(20)

        # ========================================================
        # ROW 1: WAVEFORM + FFT + PSD (side by side)
        # ========================================================

        self.row1_figure = Figure(
            figsize=(12, 3.5)
        )

        self.row1_canvas = FigureCanvas(
            self.row1_figure
        )

        self.row1_canvas.setMinimumHeight(300)
        self.row1_canvas.setFixedHeight(300)

        self.row1_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )

        analysis_layout.addWidget(
            self.row1_canvas
        )

        # ========================================================
        # 4. WATERFALL
        # ========================================================

        self.waterfall_figure = Figure(
            figsize=(10, 4)
        )

        self.waterfall_canvas = FigureCanvas(
            self.waterfall_figure
        )

        self.waterfall_canvas.setMinimumHeight(350)
        self.waterfall_canvas.setFixedHeight(350)

        self.waterfall_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )

        analysis_layout.addWidget(
            self.waterfall_canvas
        )

        # ========================================================
        # 5. CONSTELLATION
        # ========================================================

        self.constellation_figure = Figure(
            figsize=(10, 8)
        )

        self.constellation_canvas = FigureCanvas(
            self.constellation_figure
        )

        self.constellation_canvas.setMinimumHeight(700)
        self.constellation_canvas.setFixedHeight(700)

        self.constellation_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )

        analysis_layout.addWidget(
            self.constellation_canvas
        )

        # --------------------------------------------------------
        # SET ANALYSIS GROUP
        # --------------------------------------------------------

        analysis_group.setLayout(
            analysis_layout
        )

        outer_layout.addWidget(
            analysis_group
        )

        self.setLayout(
            outer_layout
        )

        # Ensure the widget requests enough height
        # so the outer scroll area can scroll to
        # the constellation diagram.
        # (300 + 350 + 700 + spacing + group padding)
        self.setMinimumHeight(1450)

    # ============================================================
    # DISPLAY ALL ANALYSIS
    # ============================================================

    def display_analysis(self, result):

        if result.get(
            "analysis_status"
        ) != "success":

            return

        self.plot_row1(
            result
        )

        self.plot_waterfall(
            result
        )

        self.plot_constellation(
            result
        )

    # ============================================================
    # ROW 1: WAVEFORM + FFT + PSD (side by side)
    # ============================================================

    def plot_row1(self, result):

        self.row1_figure.clear()

        # --------------------------------------------------------
        # 1. WAVEFORM (left)
        # --------------------------------------------------------

        ax1 = self.row1_figure.add_subplot(
            1, 3, 1
        )

        time = result[
            "waveform_time"
        ]

        i_data = result[
            "waveform_i"
        ]

        q_data = result[
            "waveform_q"
        ]

        time_ms = time * 1000

        ax1.plot(
            time_ms,
            i_data,
            label="I (In-phase)",
            linewidth=0.8
        )

        ax1.plot(
            time_ms,
            q_data,
            label="Q (Quadrature)",
            linewidth=0.8
        )

        ax1.set_title(
            "1. Time-Domain I/Q Waveform",
            fontsize=10,
            fontweight="bold"
        )

        ax1.set_xlabel(
            "Time (ms)",
            fontsize=8
        )

        ax1.set_ylabel(
            "Amplitude",
            fontsize=8
        )

        ax1.tick_params(
            labelsize=7
        )

        ax1.grid(
            True,
            alpha=0.3
        )

        ax1.legend(
            fontsize=7,
            loc="lower right"
        )

        # --------------------------------------------------------
        # 2. FFT (center)
        # --------------------------------------------------------

        ax2 = self.row1_figure.add_subplot(
            1, 3, 2
        )

        frequency = result[
            "fft_frequency"
        ]

        magnitude_db = result[
            "fft_magnitude_db"
        ]

        frequency_khz = (
            frequency / 1000
        )

        ax2.plot(
            frequency_khz,
            magnitude_db,
            linewidth=0.8
        )

        ax2.set_title(
            "2. FFT / Magnitude Spectrum",
            fontsize=10,
            fontweight="bold"
        )

        ax2.set_xlabel(
            "Frequency (kHz)",
            fontsize=8
        )

        ax2.set_ylabel(
            "Magnitude (dB)",
            fontsize=8
        )

        ax2.tick_params(
            labelsize=7
        )

        ax2.grid(
            True,
            alpha=0.3
        )

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

        ax2.scatter(
            peak_frequency / 1000,
            peak_magnitude,
            zorder=5,
            s=30
        )

        ax2.annotate(
            f"Peak: "
            f"{peak_frequency / 1000:.2f} kHz",
            (
                peak_frequency / 1000,
                peak_magnitude
            ),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=7
        )

        # --------------------------------------------------------
        # 3. PSD (right)
        # --------------------------------------------------------

        ax3 = self.row1_figure.add_subplot(
            1, 3, 3
        )

        psd_frequency = result[
            "psd_frequency"
        ]

        psd_db = result[
            "psd_db"
        ]

        psd_frequency_khz = (
            psd_frequency / 1000
        )

        ax3.plot(
            psd_frequency_khz,
            psd_db,
            linewidth=0.8
        )

        ax3.set_title(
            "3. Power Spectral Density (PSD)",
            fontsize=10,
            fontweight="bold"
        )

        ax3.set_xlabel(
            "Frequency (kHz)",
            fontsize=8
        )

        ax3.set_ylabel(
            "PSD (dB/Hz)",
            fontsize=8
        )

        ax3.tick_params(
            labelsize=7
        )

        ax3.grid(
            True,
            alpha=0.3
        )

        # --------------------------------------------------------
        # Layout
        # --------------------------------------------------------

        self.row1_figure.tight_layout(
            pad=2.0,
            w_pad=3.0
        )

        self.row1_canvas.draw()

    # ============================================================
    # 4. WATERFALL
    # ============================================================

    def plot_waterfall(self, result):

        self.waterfall_figure.clear()

        ax = self.waterfall_figure.add_subplot(
            111
        )

        # --------------------------------------------------------
        # Get signal
        # --------------------------------------------------------

        signal = np.asarray(
            result["samples"]
        )

        sample_rate = float(
            result["sample_rate"]
        )

        # --------------------------------------------------------
        # Limit signal size
        # --------------------------------------------------------

        max_samples = 500000

        if len(signal) > max_samples:

            signal = signal[
                :max_samples
            ]

        # --------------------------------------------------------
        # Spectrogram
        # --------------------------------------------------------

        nperseg = min(
            2048,
            len(signal)
        )

        if nperseg < 16:

            ax.text(
                0.5,
                0.5,
                "Not enough samples for waterfall",
                ha="center",
                va="center"
            )

            self.waterfall_canvas.draw()

            return

        frequencies, times, spectrum = spectrogram(
            signal,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            return_onesided=False,
            mode="magnitude"
        )

        # --------------------------------------------------------
        # Center frequency axis
        # --------------------------------------------------------

        frequencies = np.fft.fftshift(
            frequencies
        )

        spectrum = np.fft.fftshift(
            spectrum,
            axes=0
        )

        # --------------------------------------------------------
        # Convert to dB
        # --------------------------------------------------------

        spectrum_db = (
            20
            * np.log10(
                spectrum + 1e-12
            )
        )

        # --------------------------------------------------------
        # Plot
        # --------------------------------------------------------

        mesh = ax.pcolormesh(
            times,
            frequencies / 1000,
            spectrum_db,
            shading="auto"
        )

        ax.set_title(
            "4. Waterfall / Spectrogram",
            fontsize=13,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Time (s)"
        )

        ax.set_ylabel(
            "Frequency (kHz)"
        )

        self.waterfall_figure.colorbar(
            mesh,
            ax=ax,
            label="Power (dB)"
        )

        self.waterfall_figure.tight_layout()

        self.waterfall_canvas.draw()

    # ============================================================
    # 5. CONSTELLATION
    # ============================================================

    def plot_constellation(self, result):

        self.constellation_figure.clear()

        from matplotlib.gridspec import GridSpec

        from param_estimation.modulation_classifier import (
            get_ideal_constellation,
            MODULATION_TYPES,
            MODULATION_INFO,
        )

        # --------------------------------------------------------
        # Layout: main constellation + 5 reference types
        # --------------------------------------------------------

        gs = GridSpec(
            2,
            5,
            figure=self.constellation_figure,
            height_ratios=[2.5, 1],
            hspace=0.55,
            wspace=0.5
        )

        # ========================================================
        # MAIN CONSTELLATION (top, center 3 columns)
        # ========================================================

        ax_main = self.constellation_figure.add_subplot(
            gs[0, 1:4]
        )

        # --------------------------------------------------------
        # Get constellation data
        # --------------------------------------------------------

        if (
            "constellation_i" in result
            and "constellation_q" in result
        ):

            i_data = np.asarray(
                result["constellation_i"]
            )

            q_data = np.asarray(
                result["constellation_q"]
            )

        else:

            signal = np.asarray(
                result["samples"]
            )

            max_points = 10000

            if len(signal) > max_points:

                step = max(
                    1,
                    len(signal) // max_points
                )

                signal = signal[::step]

            i_data = np.real(signal)
            q_data = np.imag(signal)

        # --------------------------------------------------------
        # Plot signal constellation points
        # --------------------------------------------------------

        ax_main.scatter(
            i_data,
            q_data,
            s=5,
            alpha=0.6,
            label="Signal",
            zorder=3
        )

        # --------------------------------------------------------
        # Get detected modulation type
        # --------------------------------------------------------

        detected_type = result.get(
            "detected_modulation",
            "Unknown"
        )

        confidence = result.get(
            "modulation_confidence",
            0.0
        )

        # --------------------------------------------------------
        # Overlay ideal constellation points
        # --------------------------------------------------------

        if detected_type != "Unknown":

            ideal = get_ideal_constellation(
                detected_type
            )

            if ideal is not None:

                max_amp = max(
                    np.max(np.abs(i_data))
                    if len(i_data) > 0
                    else 1.0,
                    np.max(np.abs(q_data))
                    if len(q_data) > 0
                    else 1.0
                )

                if max_amp > 0:
                    ideal_scaled = ideal * max_amp
                else:
                    ideal_scaled = ideal

                ax_main.scatter(
                    np.real(ideal_scaled),
                    np.imag(ideal_scaled),
                    s=120,
                    marker="x",
                    c="red",
                    linewidths=2.5,
                    zorder=10,
                    label=(
                        f"Ideal {detected_type}"
                    )
                )

        # --------------------------------------------------------
        # Reference axes
        # --------------------------------------------------------

        ax_main.axhline(
            0,
            linewidth=0.8,
            color="gray"
        )

        ax_main.axvline(
            0,
            linewidth=0.8,
            color="gray"
        )

        # --------------------------------------------------------
        # Title with detection result
        # --------------------------------------------------------

        title_text = (
            "5. Constellation Diagram"
        )

        if detected_type != "Unknown":

            title_text += (
                f"\nDetected: {detected_type}"
                f" (Confidence:"
                f" {confidence:.0%})"
            )

        ax_main.set_title(
            title_text,
            fontsize=13,
            fontweight="bold"
        )

        ax_main.set_xlabel(
            "I (In-phase)"
        )

        ax_main.set_ylabel(
            "Q (Quadrature)"
        )

        ax_main.grid(
            True,
            alpha=0.3
        )

        ax_main.set_aspect(
            "equal",
            adjustable="box"
        )

        ax_main.legend(
            fontsize=8,
            loc="upper right"
        )

        # ========================================================
        # REFERENCE CONSTELLATION TYPES (bottom row)
        # ========================================================

        ref_colors = {
            "BPSK":   "#2196F3",
            "QPSK":   "#9C27B0",
            "8-PSK":  "#FF9800",
            "16-QAM": "#E91E63",
            "64-QAM": "#009688",
        }

        for idx, mod_type in enumerate(
            MODULATION_TYPES
        ):

            ax_ref = (
                self.constellation_figure
                .add_subplot(gs[1, idx])
            )

            ideal = get_ideal_constellation(
                mod_type
            )

            if ideal is not None:

                color = ref_colors.get(
                    mod_type,
                    "blue"
                )

                ax_ref.scatter(
                    np.real(ideal),
                    np.imag(ideal),
                    s=30,
                    c=color,
                    zorder=5
                )

            # Reference axes
            ax_ref.axhline(
                0,
                linewidth=0.5,
                color="gray",
                alpha=0.5
            )

            ax_ref.axvline(
                0,
                linewidth=0.5,
                color="gray",
                alpha=0.5
            )

            # Title with modulation info
            info = MODULATION_INFO[mod_type]

            ax_ref.set_title(
                (
                    f"{mod_type}\n"
                    f"{info['bits_per_symbol']}"
                    f" bits/symbol"
                ),
                fontsize=8,
                fontweight="bold"
            )

            ax_ref.set_aspect(
                "equal",
                adjustable="box"
            )

            ax_ref.tick_params(
                labelsize=6
            )

            ax_ref.grid(
                True,
                alpha=0.2
            )

            # Highlight detected type with
            # green border
            if mod_type == detected_type:

                for spine in (
                    ax_ref.spines.values()
                ):
                    spine.set_edgecolor(
                        "#4CAF50"
                    )
                    spine.set_linewidth(3)

        # --------------------------------------------------------
        # Final layout
        # --------------------------------------------------------

        self.constellation_figure.tight_layout()

        self.constellation_canvas.draw()

    # ============================================================
    # CLEAR GRAPHS
    # ============================================================

    def clear_plots(self):

        self.row1_figure.clear()

        self.waterfall_figure.clear()

        self.constellation_figure.clear()

        self.row1_canvas.draw()

        self.waterfall_canvas.draw()

        self.constellation_canvas.draw()