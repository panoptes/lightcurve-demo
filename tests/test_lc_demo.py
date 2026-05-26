"""Unit tests for lc_demo.py pure-Python business logic.

Streamlit session-state functions and the `main()` entrypoint are excluded
because they require a running Streamlit server; they are covered by
manual/integration testing instead.
"""

# ---------------------------------------------------------------------------
# Import the module — streamlit is imported at module level in lc_demo, so we
# mock it before the first import to avoid needing a real Streamlit context.
# ---------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Provide a minimal streamlit stub so the module loads without a server.
_st_stub = MagicMock()
sys.modules.setdefault("streamlit", _st_stub)

import lc_demo  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_frame(h: int, w: int, r: int, g: int, b: int) -> np.ndarray:
    """Return an H×W×3 uint8 RGB frame filled with a single colour."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = r
    frame[:, :, 1] = g
    frame[:, :, 2] = b
    return frame


def _gray_frame(h: int, w: int, value: int) -> np.ndarray:
    """Return an H×W uint8 greyscale frame filled with a single value."""
    return np.full((h, w), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# _measure_flux — colour mode
# ---------------------------------------------------------------------------


class TestMeasureFluxColor:
    def test_returns_three_floats(self):
        frame = _solid_frame(10, 10, 100, 150, 200)
        r, g, b = lc_demo._measure_flux(frame, use_color=True)
        assert isinstance(r, float)
        assert isinstance(g, float)
        assert isinstance(b, float)

    def test_channel_sums_correct(self):
        frame = _solid_frame(4, 4, 10, 20, 30)
        r, g, b = lc_demo._measure_flux(frame, use_color=True)
        assert r == pytest.approx(10 * 16)
        assert g == pytest.approx(20 * 16)
        assert b == pytest.approx(30 * 16)

    def test_zero_frame_gives_zeros(self):
        frame = _solid_frame(5, 5, 0, 0, 0)
        assert lc_demo._measure_flux(frame, use_color=True) == (0.0, 0.0, 0.0)

    def test_max_frame(self):
        frame = _solid_frame(2, 2, 255, 255, 255)
        r, g, b = lc_demo._measure_flux(frame, use_color=True)
        assert r == g == b == pytest.approx(255 * 4)


# ---------------------------------------------------------------------------
# _measure_flux — greyscale mode
# ---------------------------------------------------------------------------


class TestMeasureFluxGray:
    def test_all_channels_equal(self):
        frame = _gray_frame(6, 6, 80)
        r, g, b = lc_demo._measure_flux(frame, use_color=False)
        assert r == g == b

    def test_sum_correct(self):
        frame = _gray_frame(3, 3, 50)
        r, _, _ = lc_demo._measure_flux(frame, use_color=False)
        assert r == pytest.approx(50 * 9)

    def test_zero_gives_zeros(self):
        frame = _gray_frame(4, 4, 0)
        assert lc_demo._measure_flux(frame, use_color=False) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_at_baseline_returns_100(self):
        r, g, b = lc_demo._normalise(200.0, 300.0, 400.0, [200.0, 300.0, 400.0])
        assert r == pytest.approx(100.0)
        assert g == pytest.approx(100.0)
        assert b == pytest.approx(100.0)

    def test_half_flux_returns_50(self):
        r, g, b = lc_demo._normalise(50.0, 50.0, 50.0, [100.0, 100.0, 100.0])
        assert r == pytest.approx(50.0)
        assert g == pytest.approx(50.0)
        assert b == pytest.approx(50.0)

    def test_clamped_at_100(self):
        # Flux higher than baseline should not exceed 100
        r, g, b = lc_demo._normalise(200.0, 200.0, 200.0, [100.0, 100.0, 100.0])
        assert r == pytest.approx(100.0)
        assert g == pytest.approx(100.0)
        assert b == pytest.approx(100.0)

    def test_zero_reference_returns_zero(self):
        # Guard against division by zero
        r, g, b = lc_demo._normalise(50.0, 50.0, 50.0, [0.0, 0.0, 0.0])
        assert r == 0.0
        assert g == 0.0
        assert b == 0.0

    def test_mixed_channels(self):
        r, g, b = lc_demo._normalise(25.0, 75.0, 100.0, [100.0, 100.0, 100.0])
        assert r == pytest.approx(25.0)
        assert g == pytest.approx(75.0)
        assert b == pytest.approx(100.0)

    def test_zero_flux_returns_zero(self):
        r, g, b = lc_demo._normalise(0.0, 0.0, 0.0, [100.0, 100.0, 100.0])
        assert r == g == b == 0.0


# ---------------------------------------------------------------------------
# _build_figure
# ---------------------------------------------------------------------------


class TestBuildFigure:
    def _zeros(self, max_ticks: int = 50) -> np.ndarray:
        return np.zeros((3, max_ticks))

    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go

        fig = lc_demo._build_figure(self._zeros(), 50, 10, use_color=True)
        assert isinstance(fig, go.Figure)

    def test_color_mode_has_three_rgb_traces(self):
        fig = lc_demo._build_figure(self._zeros(), 50, 10, use_color=True)
        trace_names = [t.name for t in fig.data]
        assert "R" in trace_names
        assert "G" in trace_names
        assert "B" in trace_names

    def test_gray_mode_has_one_trace(self):
        fig = lc_demo._build_figure(self._zeros(), 50, 10, use_color=False)
        assert len(fig.data) == 1
        assert fig.data[0].name == "Flux"

    def test_x_axis_range_matches_lc_value(self):
        fig = lc_demo._build_figure(self._zeros(), 50, 15, use_color=True)
        assert fig.layout.xaxis.range == (0, 15)  # Plotly returns tuples

    def test_y_axis_range_fixed(self):
        fig = lc_demo._build_figure(self._zeros(), 50, 10, use_color=True)
        assert fig.layout.yaxis.range == (50, 105)  # Plotly returns tuples

    def test_time_axis_length_matches_max_ticks(self):
        max_ticks = 100
        fig = lc_demo._build_figure(self._zeros(max_ticks), max_ticks, 10, use_color=True)
        # Each trace should have max_ticks x-values
        assert len(fig.data[0].x) == max_ticks

    def test_data_reflected_in_traces(self):
        lc = np.zeros((3, 10))
        lc[0, 5] = 80.0  # R channel spike at tick 5
        fig = lc_demo._build_figure(lc, 10, 5, use_color=True)
        r_trace = next(t for t in fig.data if t.name == "R")
        assert r_trace.y[5] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# _get_frame — tested with a mock VideoCapture
# ---------------------------------------------------------------------------


class TestGetFrame:
    def _make_cap(self, h: int = 120, w: int = 160) -> MagicMock:
        """Return a mock cv2.VideoCapture that yields a solid blue BGR frame."""
        bgr_frame = np.zeros((h, w, 3), dtype=np.uint8)
        bgr_frame[:, :, 0] = 200  # blue channel in BGR
        cap = MagicMock()
        cap.read.return_value = (True, bgr_frame)
        return cap

    def test_display_is_rgb_and_correct_shape(self):
        cap = self._make_cap(120, 160)
        display, _ = lc_demo._get_frame(cap, radius=40, use_color=True)
        assert display.shape == (120, 160, 3)

    def test_color_photometry_is_3channel(self):
        cap = self._make_cap(120, 160)
        _, photometry = lc_demo._get_frame(cap, radius=40, use_color=True)
        assert photometry.ndim == 3
        assert photometry.shape[2] == 3

    def test_gray_photometry_is_2channel(self):
        cap = self._make_cap(120, 160)
        _, photometry = lc_demo._get_frame(cap, radius=40, use_color=False)
        assert photometry.ndim == 2

    def test_display_has_aperture_circle_drawn(self):
        """The yellow aperture circle should add non-zero pixels to display."""
        cap = self._make_cap(120, 160)
        # Use a black frame so any circle drawing is detectable
        cap.read.return_value = (True, np.zeros((120, 160, 3), dtype=np.uint8))
        display, _ = lc_demo._get_frame(cap, radius=40, use_color=True)
        # Yellow = (255, 255, 0) in RGB — at least some pixels should be non-zero
        assert display.sum() > 0

    def test_mask_zeros_outside_aperture(self):
        """Pixels outside the circular aperture should be zeroed in photometry."""
        h, w = 100, 100
        # Solid white frame — all pixels lit
        white_bgr = np.full((h, w, 3), 255, dtype=np.uint8)
        cap = MagicMock()
        cap.read.return_value = (True, white_bgr)
        _, photometry = lc_demo._get_frame(cap, radius=5, use_color=True)
        # With radius=5, centre patch is bright; corners (far from centre) must be 0
        assert photometry[0, 0, 0] == 0  # top-left corner zeroed
        assert photometry[99, 99, 0] == 0  # bottom-right corner zeroed

    def test_failed_read_calls_st_stop(self):
        cap = MagicMock()
        cap.read.return_value = (False, None)
        # The real st.stop() raises StopException; give our stub the same behaviour
        # so _get_frame doesn't continue past the error branch.
        _st_stub.stop.side_effect = SystemExit
        with pytest.raises(SystemExit):
            lc_demo._get_frame(cap, radius=40, use_color=True)
        _st_stub.stop.side_effect = None  # reset for other tests

    def test_radius_clamp_to_minimum_one(self):
        """A radius of 0 should not crash — it is clamped to 1."""
        cap = self._make_cap()
        display, _ = lc_demo._get_frame(cap, radius=0, use_color=True)
        assert display is not None
