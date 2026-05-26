#!/usr/bin/env python3
"""Lightcurve Demo — interactive web exhibit powered by Streamlit.

Point a webcam at a light source, click Start, and move a dark object in
front of the light to see a real-time lightcurve appear in your browser.
"""

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICK_MS: int = (
    100  # update interval in milliseconds (10 fps); must be ≥100 for st.fragment run_every
)
TICK_S: float = TICK_MS / 1000.0
NORM_MAX: float = 100.0  # normalised light value at baseline

# Python 3.12 type aliases
type FluxTriple = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _init_state() -> None:
    """Initialise all session-state keys on first run."""
    defaults: dict[str, object] = {
        "lc_active": False,
        "video_active": True,  # webcam feed running
        "lc_value": 10,  # total duration (seconds)
        "tick_num": 0,
        "lc_data": None,  # numpy array shape (3, max_ticks)
        "normal_factor": [1.0, 1.0, 1.0],
        "cap": None,  # cv2.VideoCapture
        "video_device": 0,
        "frame_rgb": None,  # latest raw RGB frame
        "masked_rgb": None,  # latest masked RGB frame (for display)
        "image_bytes": None,  # PNG bytes for download button
        "lc_fig": None,  # current Plotly figure
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_lc() -> None:
    """Reset lightcurve data arrays for a new recording."""
    max_ticks = int((st.session_state.lc_value * 1000) / TICK_MS)
    st.session_state.tick_num = 0
    st.session_state.lc_data = np.zeros((3, max_ticks))
    st.session_state.normal_factor = [1.0, 1.0, 1.0]
    st.session_state.image_bytes = None
    st.session_state.lc_fig = None


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def _open_camera(device: int) -> cv2.VideoCapture:
    """Open (or re-open) the video capture device."""
    cap = st.session_state.cap
    if cap is not None and cap.isOpened():
        return cap
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        st.error(f"Cannot open video device {device}. Check the device index in the sidebar.")
        st.stop()
    st.session_state.cap = cap
    return cap


def _release_camera() -> None:
    """Release the video capture device."""
    cap = st.session_state.cap
    if cap is not None and cap.isOpened():
        cap.release()
    st.session_state.cap = None


def _get_frame(
    cap: cv2.VideoCapture, radius: int, use_color: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Read one frame and return (display_rgb, photometry_data).

    display_rgb  — full-frame RGB image with the circular aperture overlaid.
    photometry_data — masked array used for flux measurement.
                      RGB (H×W×3) when use_color=True, greyscale (H×W) otherwise.
    """
    ret, frame = cap.read()
    if not ret:
        st.error("Lost connection to camera.")
        st.stop()

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Build circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), max(1, radius), 1, -1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw aperture circle on display image
    display = frame_rgb.copy()
    cv2.circle(display, (cx, cy), max(1, radius), (255, 255, 0), 2)

    if use_color:
        photometry = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        photometry = cv2.bitwise_and(gray, gray, mask=mask)

    return display, photometry


# ---------------------------------------------------------------------------
# Photometry helpers
# ---------------------------------------------------------------------------


def _measure_flux(photometry: np.ndarray, use_color: bool) -> FluxTriple:
    """Return (r, g, b) flux sums.  All three are equal for greyscale."""
    if use_color:
        r = float(photometry[:, :, 0].sum())
        g = float(photometry[:, :, 1].sum())
        b = float(photometry[:, :, 2].sum())
    else:
        v = float(photometry.sum())
        r = g = b = v
    return r, g, b


def _normalise(r: float, g: float, b: float, factor: list[float]) -> FluxTriple:
    """Normalise flux values to baseline=100, clamped at 100."""

    def safe_div(val: float, ref: float) -> float:
        if ref == 0:
            return 0.0
        return min(val / ref * NORM_MAX, NORM_MAX)

    return safe_div(r, factor[0]), safe_div(g, factor[1]), safe_div(b, factor[2])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _build_figure(
    lc_data: np.ndarray,
    max_ticks: int,
    lc_value: int,
    use_color: bool,
    filled_ticks: int | None = None,
) -> go.Figure:
    """Build a Plotly figure from the current lightcurve data.

    filled_ticks — number of data points that have been written.  Only those
                   points are plotted; the rest of the pre-allocated array
                   (zeros) is intentionally omitted so the chart grows in
                   real time instead of appearing blank.  Pass None (default)
                   to plot the full array.
    """
    n = filled_ticks if filled_ticks is not None else max_ticks
    t = np.arange(max_ticks) * TICK_S

    fig = go.Figure()
    fig.add_hline(y=NORM_MAX, line_dash="dash", line_color="gray", opacity=0.5)

    if use_color:
        for i, (color, name) in enumerate(zip(["red", "green", "blue"], ["R", "G", "B"])):
            fig.add_trace(
                go.Scatter(
                    x=t[:n],
                    y=lc_data[i, :n],
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 2},
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=t[:n],
                y=lc_data[0, :n],
                mode="lines+markers",
                marker={"size": 4},
                name="Flux",
                line={"color": "gray", "width": 2},
            )
        )

    fig.update_layout(
        xaxis_title="Time [s]",
        yaxis_title="Light [%]",
        yaxis={"range": [0, 105]},
        xaxis={"range": [0, lc_value]},
        margin={"l": 50, "r": 20, "t": 20, "b": 50},
        legend={"orientation": "h", "y": 1.05},
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# Live view fragment
# ---------------------------------------------------------------------------


@st.fragment(run_every=TICK_S)
def _live_view(radius: int, use_color: bool, save_image: bool) -> None:
    """Camera acquisition and lightcurve display, auto-refreshed every TICK_S.

    Running as a fragment means only this portion of the DOM is re-rendered
    each tick; the sidebar and page chrome remain perfectly still.
    """
    lc_value: int = st.session_state.lc_value
    max_ticks: int = int((lc_value * 1000) / TICK_MS)

    col_cam, col_lc = st.columns([1, 2])

    with col_cam:
        st.subheader("Webcam")
        if st.session_state.video_active:
            cap = _open_camera(st.session_state.video_device)
            display, photometry = _get_frame(cap, radius, use_color)
            st.image(display, channels="RGB", width="stretch")
        else:
            st.info("📷 Video stopped. Press **Start Video** in the sidebar to resume.")
            photometry = None

    with col_lc:
        st.subheader("Lightcurve")

        if st.session_state.lc_active and photometry is not None:
            # Pin max_ticks to the array allocated at recording start so that
            # changing the duration slider mid-recording cannot cause an IndexError.
            lc = st.session_state.lc_data
            max_ticks = lc.shape[1]
            tick = st.session_state.tick_num

            elapsed = tick * TICK_S
            remaining = max(0.0, lc_value - elapsed)
            st.caption(f"Recording… {remaining:.1f}s remaining")

            r, g, b = _measure_flux(photometry, use_color)

            # First tick — capture baseline
            if tick == 0:
                factor = [r if r > 0 else 1.0, g if g > 0 else 1.0, b if b > 0 else 1.0]
                st.session_state.normal_factor = factor
                if save_image:
                    _, enc = cv2.imencode(".png", cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
                    st.session_state.image_bytes = enc.tobytes()

            # Midpoint snapshot
            mid = max_ticks // 2
            if save_image and tick == mid:
                _, enc = cv2.imencode(".png", cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
                st.session_state.image_bytes = enc.tobytes()

            nr, ng, nb = _normalise(r, g, b, st.session_state.normal_factor)
            lc[0, tick] = nr
            lc[1, tick] = ng
            lc[2, tick] = nb

            # Plot only the filled portion so the chart grows in real time.
            fig = _build_figure(lc, max_ticks, lc_value, use_color, filled_ticks=tick + 1)
            st.plotly_chart(fig, width="stretch", key="lc_chart")

            st.session_state.tick_num += 1

            if st.session_state.tick_num >= max_ticks:
                st.session_state.lc_active = False
                st.session_state.lc_fig = _build_figure(lc, max_ticks, lc_value, use_color)
                # Full rerun to restore the sidebar Start/Clear button states.
                st.rerun()

        else:
            if st.session_state.lc_fig is not None:
                st.plotly_chart(st.session_state.lc_fig, width="stretch", key="lc_chart")
            else:
                empty_fig = _build_figure(np.zeros((3, max_ticks)), max_ticks, lc_value, use_color)
                st.plotly_chart(empty_fig, width="stretch", key="lc_chart")

            if st.session_state.image_bytes is not None:
                st.download_button(
                    label="⬇ Download snapshot",
                    data=st.session_state.image_bytes,
                    file_name="lightcurve_snapshot.png",
                    mime="image/png",
                )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Lightcurve Demo",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()

    # ------------------------------------------------------------------
    # Sidebar controls
    # ------------------------------------------------------------------
    with st.sidebar:
        st.title("🔭 Lightcurve Demo")
        st.caption(
            "Point your webcam at a light source, press **Start**, then move a dark object in front of it."
        )

        st.divider()

        # Video start/stop
        if st.session_state.video_active:
            if st.button("⏹ Stop Video", use_container_width=True):
                _release_camera()
                st.session_state.video_active = False
                if st.session_state.lc_active:
                    st.session_state.lc_active = False
                    _reset_lc()
                # Force a full rerun so the sidebar re-renders with the correct button.
                st.rerun()
        else:
            if st.button("📷 Start Video", use_container_width=True):
                st.session_state.video_active = True
                # Force a full rerun so the sidebar re-renders with the correct button.
                st.rerun()

        # Device picker — collapsed in a popover to keep sidebar clean
        device_label = f"📷 Device: /dev/video{st.session_state.video_device}"
        with st.popover(device_label, use_container_width=True):
            new_device = st.number_input(
                "Video device index",
                min_value=0,
                max_value=9,
                value=st.session_state.video_device,
                step=1,
                help="0 = /dev/video0, 1 = /dev/video1, …",
            )
            if new_device != st.session_state.video_device:
                _release_camera()
                st.session_state.video_device = new_device

        st.divider()

        lc_value = st.slider(
            "Duration (seconds)",
            min_value=2,
            max_value=30,
            value=st.session_state.lc_value,
            step=1,
        )
        st.session_state.lc_value = lc_value

        radius = st.slider(
            "Aperture radius (px)",
            min_value=5,
            max_value=200,
            value=100,
            step=5,
            help="Size of the circular region used to measure light.",
        )

        use_color = st.toggle("Show colour channels", value=True)

        save_image = st.toggle("Save snapshot at midpoint", value=False)

        st.divider()

        col_start, col_clear = st.columns(2)
        with col_start:
            start_pressed = st.button(
                "▶ Start",
                use_container_width=True,
                disabled=st.session_state.lc_active or not st.session_state.video_active,
            )
        with col_clear:
            clear_pressed = st.button(
                "✕ Clear",
                use_container_width=True,
                disabled=not st.session_state.lc_active,
            )

        if start_pressed and not st.session_state.lc_active:
            _reset_lc()
            st.session_state.lc_active = True
            # Force a full rerun so the sidebar re-renders with the Start button
            # disabled.  Without this, lc_active is set True *after* the button
            # has already been rendered as enabled in the current script run.
            st.rerun()

        if clear_pressed:
            st.session_state.lc_active = False
            _reset_lc()

    _live_view(radius=radius, use_color=use_color, save_image=save_image)


if __name__ == "__main__":
    main()
