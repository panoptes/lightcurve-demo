#!/usr/bin/env python3
"""Lightcurve Demo — interactive web exhibit powered by Streamlit.

Point a webcam at a light source, click Start, and move a dark object in
front of the light to see a real-time lightcurve appear in your browser.
"""

import io
from datetime import datetime

import cv2
import matplotlib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

matplotlib.use("Agg")
from matplotlib import pyplot as plt

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
        "frame_at_midpoint": None,  # RGB numpy array captured at midpoint for snapshot
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
    st.session_state.frame_at_midpoint = None
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
    zoom_yaxis: bool = True,
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
        yaxis={"range": [50, 105] if zoom_yaxis else [0, 105]},
        xaxis={"range": [0, lc_value]},
        margin={"l": 50, "r": 20, "t": 20, "b": 50},
        legend={"orientation": "h", "y": 1.05},
        template="plotly_dark",
    )
    return fig


def _build_composite_image(
    frame_rgb: np.ndarray,
    lc_data: np.ndarray,
    max_ticks: int,
    lc_value: int,
    use_color: bool,
) -> bytes:
    """Compose a shareable PNG: webcam midpoint frame + full lightcurve + timestamp."""
    bg = "#0e1117"
    fg = "#fafafa"
    panel_bg = "#1a1d27"

    fig = plt.figure(figsize=(14, 6.5), facecolor=bg)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.10, 0.90],
        width_ratios=[1, 1.6],
        hspace=0.06,
        wspace=0.10,
        left=0.04,
        right=0.97,
        top=0.93,
        bottom=0.10,
    )

    # Header — title and timestamp
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(bg)
    ax_hdr.axis("off")
    ts = datetime.now().strftime("%d %B %Y — %H:%M")
    ax_hdr.text(
        0.5,
        0.5,
        f"PANOPTES Lightcurve Demo   •   {ts}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=fg,
        transform=ax_hdr.transAxes,
    )

    # Webcam panel
    ax_cam = fig.add_subplot(gs[1, 0])
    ax_cam.set_facecolor(panel_bg)
    ax_cam.imshow(frame_rgb)
    ax_cam.set_title("Webcam (midpoint)", color=fg, fontsize=11, pad=6)
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])
    for spine in ax_cam.spines.values():
        spine.set_edgecolor("#444444")

    # Lightcurve panel
    ax_lc = fig.add_subplot(gs[1, 1])
    ax_lc.set_facecolor(panel_bg)
    t = np.arange(max_ticks) * TICK_S

    if use_color:
        palette = [("#ef553b", "R"), ("#00cc96", "G"), ("#636efa", "B")]
        for i, (color, label) in enumerate(palette):
            ax_lc.plot(t, lc_data[i], color=color, linewidth=2, label=label)
        ax_lc.legend(
            loc="lower left",
            facecolor=panel_bg,
            edgecolor="#444444",
            labelcolor=fg,
            fontsize=9,
        )
    else:
        ax_lc.plot(t, lc_data[0], color="#aaaaaa", linewidth=2, label="Flux")

    ax_lc.axhline(NORM_MAX, linestyle="--", color="#666666", linewidth=1, alpha=0.7)
    ax_lc.autoscale(axis="y")
    ax_lc.set_xlim(0, lc_value)
    ax_lc.set_xlabel("Time [s]", color=fg, fontsize=10)
    ax_lc.set_ylabel("Light [%]", color=fg, fontsize=10)
    ax_lc.set_title("Lightcurve", color=fg, fontsize=11, pad=6)
    ax_lc.tick_params(colors=fg)
    ax_lc.xaxis.label.set_color(fg)
    ax_lc.yaxis.label.set_color(fg)
    for spine in ax_lc.spines.values():
        spine.set_edgecolor("#444444")

    # Footer tagline
    fig.text(
        0.5,
        0.02,
        "You just observed a stellar transit!  ★  projectpanoptes.org",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#aaaaaa",
        style="italic",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=bg, dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Live view fragment
# ---------------------------------------------------------------------------


@st.fragment(run_every=TICK_S)
def _live_view(radius: int, use_color: bool, save_image: bool, zoom_yaxis: bool) -> None:
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

        # Reserve fixed DOM slots upfront so the element tree never changes shape
        # regardless of recording state or which toggles are active.  Streamlit
        # fragment reconciliation is sensitive to structural position changes;
        # locking every dynamic element into its own st.empty() slot prevents the
        # chart from flickering or disappearing when the caption or download button
        # appears/disappears.
        caption_slot = st.empty()
        chart_slot = st.empty()
        download_slot = st.empty()

        if st.session_state.lc_active and photometry is not None:
            # Pin max_ticks to the array allocated at recording start so that
            # changing the duration slider mid-recording cannot cause an IndexError.
            lc = st.session_state.lc_data
            max_ticks = lc.shape[1]
            tick = st.session_state.tick_num

            elapsed = tick * TICK_S
            remaining = max(0.0, lc_value - elapsed)
            caption_slot.caption(f"Recording… {remaining:.1f}s remaining")

            r, g, b = _measure_flux(photometry, use_color)

            # First tick — capture baseline
            if tick == 0:
                factor = [r if r > 0 else 1.0, g if g > 0 else 1.0, b if b > 0 else 1.0]
                st.session_state.normal_factor = factor

            # Midpoint — store webcam frame for the composite snapshot
            mid = max_ticks // 2
            if save_image and tick == mid:
                st.session_state.frame_at_midpoint = display

            nr, ng, nb = _normalise(r, g, b, st.session_state.normal_factor)
            lc[0, tick] = nr
            lc[1, tick] = ng
            lc[2, tick] = nb

            st.session_state.lc_fig = _build_figure(
                lc, max_ticks, lc_value, use_color, filled_ticks=tick + 1, zoom_yaxis=zoom_yaxis
            )

            st.session_state.tick_num += 1

            if st.session_state.tick_num >= max_ticks:
                st.session_state.lc_active = False
                if save_image and st.session_state.frame_at_midpoint is not None:
                    st.session_state.image_bytes = _build_composite_image(
                        st.session_state.frame_at_midpoint,
                        lc,
                        max_ticks,
                        lc_value,
                        use_color,
                    )
                # Full rerun to restore the sidebar Start/Clear button states.
                st.rerun()

        # Always render the chart in its reserved slot.
        if st.session_state.lc_fig is not None:
            chart_slot.plotly_chart(st.session_state.lc_fig, width="stretch")
        else:
            chart_slot.plotly_chart(
                _build_figure(
                    np.zeros((3, max_ticks)), max_ticks, lc_value, use_color, zoom_yaxis=zoom_yaxis
                ),
                width="stretch",
            )

        if st.session_state.image_bytes is not None:
            download_slot.download_button(
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

        zoom_yaxis = st.toggle("Zoom Y-axis (50–100%)", value=True)

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
                disabled=st.session_state.lc_active or st.session_state.lc_fig is None,
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

    _live_view(radius=radius, use_color=use_color, save_image=save_image, zoom_yaxis=zoom_yaxis)


if __name__ == "__main__":
    main()
