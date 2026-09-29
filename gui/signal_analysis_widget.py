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
import warnings
import numpy as np
from scipy.signal import spectrogram

from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QScrollArea,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas
)

from matplotlib.figure import Figure

from param_estimation import estimate_symbol_rate
from param_estimation.symbol_rate_estimator import SymbolRateEstimator
from param_estimation.modulation_classifier import ModulationClassifier


class SignalAnalysisWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_result = None
        self.pipeline_result = None

        self.setup_ui()

    def _safe_tight_layout(self, figure):
        """
        Safely calls tight_layout() on a Matplotlib figure while suppressing
        UserWarning warnings when Axes are incompatible with tight_layout.
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                figure.tight_layout()
        except Exception:
            pass

    # ============================================================
    # MATPLOTLIB DARK-THEME HELPERS
    # ============================================================

    def _style_figure(self, figure):
        """Keep Matplotlib figures readable inside the dark Qt UI."""
        figure.patch.set_facecolor("#1e1e1e")

    def _style_axes(self, ax):
        """Apply consistent dark styling without changing plot data."""
        ax.set_facecolor("#1e1e1e")

        ax.title.set_color("#e0e0e0")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")

        ax.tick_params(
            axis="both",
            colors="#cccccc",
            labelsize=9,
        )

        for spine in ax.spines.values():
            spine.set_color("#555555")

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
        self._style_figure(self.waveform_figure)

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
        self._style_figure(self.fft_figure)

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
        self._style_figure(self.psd_figure)

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
        self._style_figure(self.waterfall_figure)

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

        # Keep the original compact constellation size.
        # QScrollArea only becomes scrollable when this fixed-size
        # canvas is larger than the visible tab area.
        self.constellation_figure = Figure(
            figsize=(10, 7),
            dpi=100
        )
        self._style_figure(self.constellation_figure)

        self.constellation_canvas = FigureCanvas(
            self.constellation_figure
        )

        # Fixed 1000 x 700 px canvas: do not enlarge the graph just
        # to force scrolling. The scrollbars appear only when needed.
        self.constellation_canvas.setMinimumSize(1000, 700)
        self.constellation_canvas.setMaximumSize(1000, 700)
        self.constellation_canvas.resize(1000, 700)

        self.constellation_scroll_area = QScrollArea()
        self.constellation_scroll_area.setWidgetResizable(False)
        self.constellation_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.constellation_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.constellation_scroll_area.setWidget(
            self.constellation_canvas
        )

        # Normal mouse wheel = vertical scrolling.
        # Shift + mouse wheel = horizontal scrolling.
        # QScrollArea already supports dragging/clicking its scrollbars.
        self.constellation_scroll_area.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )

        constellation_layout = QVBoxLayout(
            self.constellation_page
        )
        constellation_layout.setContentsMargins(0, 0, 0, 0)
        constellation_layout.setSpacing(0)

        constellation_layout.addWidget(
            self.constellation_scroll_area
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
    # PIPELINE RESULT INTEGRATION
    # ============================================================

    def set_pipeline_result(self, pipeline_result):
        """Store the authoritative full-pipeline result for display."""
        self.pipeline_result = pipeline_result or {}

    # ============================================================
    # PROCESS SIGNAL & DISPLAY (PIPELINE CALL)
    # ============================================================

    def process_and_update_gui(self, signal_data, sample_rate, peak_freq=7500.0):
        """
        Runs signal analysis pipelines and updates all GUI visual displays.
        """
        signal_data = np.asarray(signal_data)
        
        # 1. Modulation classification
        mod_classifier = ModulationClassifier()
        mod_type, confidence = mod_classifier.classify_modulation(signal_data, sample_rate)
        
        # 2. Symbol rate estimation (Auto Estimation via Package Function / Class)
        try:
            symbol_rate = estimate_symbol_rate(signal_data, sample_rate)
        except Exception:
            estimator = SymbolRateEstimator()
            symbol_rate = estimator.estimate_symbol_rate(signal_data, sample_rate)

        # 3. Construct standard result package
        result = {
            "analysis_status": "success",
            "samples": signal_data,
            "sample_rate": sample_rate,
            "peak_frequency": peak_freq,
            "detected_modulation": mod_type,
            "modulation_confidence": confidence,
            "estimated_symbol_rate": symbol_rate
        }

        self.display_analysis(result)

    # ============================================================
    # DISPLAY ALL ANALYSIS
    # ============================================================

    def display_analysis(self, result):

        if result.get(
            "analysis_status"
        ) != "success":

            return

        # Preserve the independent GUI/visual estimate.
        if "constellation_modulation" not in result:
            result["constellation_modulation"] = result.get(
                "detected_modulation",
                result.get("modulation", "Unknown")
            )
            result["constellation_confidence"] = result.get(
                "modulation_confidence",
                result.get("confidence", 0.0)
            )

        # The full pipeline is authoritative when supplied.
        pipeline = result.get("pipeline_result") or self.pipeline_result

        if pipeline:
            pipeline_mod = pipeline.get(
                "modulation_type",
                pipeline.get("modulation")
            )
            pipeline_conf = pipeline.get(
                "confidence",
                pipeline.get("modulation_confidence")
            )

            if pipeline_mod:
                result["pipeline_modulation"] = pipeline_mod
                result["pipeline_confidence"] = pipeline_conf
                result["pipeline_modulation_source"] = pipeline.get(
                    "modulation_source",
                    "classifier"
                )
                result["detected_modulation"] = pipeline_mod
                result["modulation_confidence"] = pipeline_conf

        self.current_result = result

        self.plot_waveform(result)
        self.plot_fft(result)
        self.plot_psd(result)
        self.plot_waterfall(result)
        self.plot_constellation(result)

    # ============================================================
    # 1. WAVEFORM
    # ============================================================

    def plot_waveform(self, result):

        self.waveform_figure.clear()

        ax = self.waveform_figure.add_subplot(111)
        self._style_axes(ax)

        signal = np.asarray(
            result["samples"]
        )

        sample_rate = float(
            result["sample_rate"]
        )

        max_points = 5000
        original_len = len(signal)

        if original_len > max_points:
            step = max(1, original_len // max_points)
            indices = np.arange(0, original_len, step)
            signal = signal[indices]
            time = indices / sample_rate
        else:
            time = np.arange(original_len) / sample_rate

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

        self.waveform_figure.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.86)

        self.waveform_canvas.draw()

    # ============================================================
    # 2. FFT / MAGNITUDE SPECTRUM
    # ============================================================

    def plot_fft(self, result):

        self.fft_figure.clear()

        ax = self.fft_figure.add_subplot(111)
        self._style_axes(ax)

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

        self.fft_figure.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.86)

        self.fft_canvas.draw()

    # ============================================================
    # 3. PSD
    # ============================================================

    def plot_psd(self, result):

        self.psd_figure.clear()

        ax = self.psd_figure.add_subplot(111)
        self._style_axes(ax)

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

        self.psd_figure.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.86)

        self.psd_canvas.draw()

    # ============================================================
    # 4. WATERFALL / SPECTROGRAM
    # ============================================================

    def plot_waterfall(self, result):

        self.waterfall_figure.clear()

        ax = self.waterfall_figure.add_subplot(111)
        self._style_axes(ax)

        signal = np.asarray(
            result["samples"]
        )

        sample_rate = float(
            result["sample_rate"]
        )

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

        colorbar = self.waterfall_figure.colorbar(
            mesh,
            ax=ax,
            label="Magnitude (dB)"
        )

        colorbar.ax.tick_params(colors="#cccccc")
        colorbar.set_label("Magnitude (dB)", color="#cccccc")

        for spine in colorbar.ax.spines.values():
            spine.set_color("#555555")

        ax.set_title(
            "Waterfall / Spectrogram",
            color="#e0e0e0",
            pad=10,
        )

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "Frequency (Hz)"
        )

        self.waterfall_figure.subplots_adjust(left=0.075, right=0.91, bottom=0.13, top=0.84)

        self.waterfall_canvas.draw()

    # ============================================================
    # 5. CONSTELLATION
    # ============================================================

    def plot_constellation(self, result):

        from param_estimation.modulation_classifier import (
            get_ideal_constellation,
            MODULATION_INFO,
        )

        self.constellation_figure.clear()
        self._style_figure(self.constellation_figure)

        signal = np.asarray(result["samples"])

        if len(signal) == 0:
            ax = self.constellation_figure.add_axes(
                [0.30, 0.24, 0.40, 0.55]
            )
            self._style_axes(ax)
            ax.set_title(
                "I/Q Constellation Diagram",
                color="#e0e0e0",
                pad=10,
            )
            ax.text(
                0.5,
                0.5,
                "No signal samples available",
                ha="center",
                va="center",
                color="#cccccc",
                transform=ax.transAxes,
            )
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

        amplitude = np.max(np.abs(signal))

        if amplitude > 0:
            i_data = i_data / amplitude
            q_data = q_data / amplitude

        # --------------------------------------------------------
        # MODULATION INFO
        # --------------------------------------------------------

        pipeline_modulation = result.get("pipeline_modulation")
        pipeline_confidence = result.get("pipeline_confidence")

        visual_modulation = result.get(
            "constellation_modulation",
            result.get(
                "detected_modulation",
                result.get("modulation", "Unknown")
            )
        )
        visual_confidence = result.get(
            "constellation_confidence",
            result.get(
                "modulation_confidence",
                result.get("confidence", 0.0)
            )
        )

        def _confidence_percent(value):
            if isinstance(value, (int, float, np.number)):
                value = float(value)
                if value <= 1:
                    value *= 100
                return int(round(value))
            return 0

        pipeline_pct = _confidence_percent(pipeline_confidence)
        visual_pct = _confidence_percent(visual_confidence)

        # --------------------------------------------------------
        # COMPACT FIXED LAYOUT
        #
        # The canvas remains 1000 x 700 px, matching the original
        # figure size. Nothing is enlarged to create scrolling.
        #
        # If the user's visible tab is smaller than 1000 x 700,
        # QScrollArea automatically shows the appropriate scrollbar.
        # --------------------------------------------------------

        if pipeline_modulation:
            title = (
                f"I/Q Constellation — "
                f"ML Pipeline: {pipeline_modulation} "
                f"({pipeline_pct}% confidence)"
            )

            if (
                visual_modulation
                and visual_modulation != pipeline_modulation
            ):
                title += (
                    f"  |  Visual estimate: {visual_modulation} "
                    f"({visual_pct}%)"
                )
        elif visual_modulation != "Unknown":
            title = (
                f"I/Q Constellation — "
                f"Visual estimate: {visual_modulation} "
                f"({visual_pct}% confidence)"
            )
        else:
            title = "I/Q Constellation"

        self.constellation_figure.suptitle(
            title,
            fontsize=14,
            fontweight="bold",
            color="#e0e0e0",
            y=0.965,
        )

        # ========================================================
        # MAIN I/Q CONSTELLATION
        # ========================================================

        # Compact centered plot.
        ax_main = self.constellation_figure.add_axes(
            [0.37, 0.43, 0.26, 0.42]
        )
        self._style_axes(ax_main)

        ax_main.scatter(
            i_data,
            q_data,
            s=9,
            alpha=0.45,
            color="#4fc3f7",
            label="Signal",
            zorder=2,
            rasterized=True,
        )

        ideal_points = get_ideal_constellation(visual_modulation)

        if ideal_points is not None:
            ax_main.scatter(
                np.real(ideal_points),
                np.imag(ideal_points),
                s=110,
                marker="X",
                color="#e53935",
                linewidths=1.0,
                edgecolors="#b71c1c",
                label=f"Visual ideal {visual_modulation}",
                zorder=3,
            )

        ax_main.axhline(
            0,
            linewidth=0.7,
            color="#777777",
            zorder=1,
        )
        ax_main.axvline(
            0,
            linewidth=0.7,
            color="#777777",
            zorder=1,
        )

        max_abs = max(
            1.15,
            float(
                max(
                    np.max(np.abs(i_data)),
                    np.max(np.abs(q_data)),
                )
            ) * 1.10,
        )

        ax_main.set_xlim(-max_abs, max_abs)
        ax_main.set_ylim(-max_abs, max_abs)
        ax_main.set_aspect("equal", adjustable="box")

        ax_main.set_xlabel(
            "I (In-phase)",
            color="#cccccc",
            labelpad=4,
            fontsize=9,
        )

        ax_main.set_ylabel(
            "Q (Quadrature)",
            color="#cccccc",
            labelpad=4,
            fontsize=9,
        )

        ax_main.tick_params(
            colors="#cccccc",
            labelsize=8,
            pad=2,
        )

        ax_main.grid(
            True,
            alpha=0.22,
            linewidth=0.6,
            color="#888888",
        )

        legend = ax_main.legend(
            loc="upper right",
            fontsize=7,
            framealpha=0.80,
            borderpad=0.4,
            handlelength=1.2,
        )

        if legend is not None:
            legend.get_frame().set_facecolor("#2b2b2b")
            legend.get_frame().set_edgecolor("#666666")

            for legend_text in legend.get_texts():
                legend_text.set_color("#e0e0e0")

        # ========================================================
        # REFERENCE CONSTELLATIONS
        # ========================================================

        ref_types = [
            "BPSK",
            "QPSK",
            "8-PSK",
            "16-QAM",
            "64-QAM",
        ]

        ref_colors = [
            "#42a5f5",
            "#ab47bc",
            "#ff9800",
            "#e91e63",
            "#26a69a",
        ]

        # Five compact, evenly spaced reference plots.
        centers = [
            0.10,
            0.30,
            0.50,
            0.70,
            0.90,
        ]

        for col, mod_name in enumerate(ref_types):

            ax_ref = self.constellation_figure.add_axes(
                [
                    centers[col] - 0.055,
                    0.08,
                    0.11,
                    0.20,
                ]
            )

            self._style_axes(ax_ref)

            pts = get_ideal_constellation(mod_name)

            if pts is not None:
                ax_ref.scatter(
                    np.real(pts),
                    np.imag(pts),
                    s=25,
                    color=ref_colors[col],
                    zorder=2,
                )

            info = MODULATION_INFO.get(
                mod_name,
                {}
            )

            bps = info.get(
                "bits_per_symbol",
                "?"
            )

            ax_ref.set_title(
                f"{mod_name}\n{bps} bits/symbol",
                fontsize=8,
                fontweight="bold",
                color="#e0e0e0",
                pad=5,
            )

            ax_ref.set_xlim(-1.3, 1.3)
            ax_ref.set_ylim(-1.3, 1.3)
            ax_ref.set_aspect(
                "equal",
                adjustable="box"
            )

            ax_ref.tick_params(
                labelsize=6,
                colors="#aaaaaa",
                pad=1,
            )

            ax_ref.grid(
                True,
                alpha=0.15,
                linewidth=0.5,
                color="#888888",
            )

            selected_modulation = pipeline_modulation or visual_modulation

            if mod_name == selected_modulation:

                for spine in ax_ref.spines.values():
                    spine.set_edgecolor("#4caf50")
                    spine.set_linewidth(2.5)

            else:

                for spine in ax_ref.spines.values():
                    spine.set_edgecolor("#555555")
                    spine.set_linewidth(0.7)

        # No tight_layout() here.
        # All positions are deliberately fixed so the plots cannot
        # overlap. QScrollArea handles any lack of screen space.
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

            max_points = 100000
            original_len = len(signal)

            if original_len > max_points:
                step = max(1, original_len // max_points)
                indices = np.arange(0, original_len, step)
                signal = signal[indices]
                time = indices / sample_rate
            else:
                time = np.arange(original_len) / sample_rate

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
    # SAVE PLOT
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
