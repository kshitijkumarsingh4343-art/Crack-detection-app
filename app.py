import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from utils import (
    segment_crack,
    connected_crack_components,
    measure_crack,
    crack_density_map,
    IS_456_CRACK_LIMITS_MM,
)

st.set_page_config(page_title="Structural Crack Inspection", layout="wide")
st.title("Structural Crack Inspection")

if "points" not in st.session_state:
    st.session_state.points = []
if "last_click" not in st.session_state:
    st.session_state.last_click = None

uploaded = st.file_uploader("Upload Structural Surface Image", type=["jpg", "jpeg", "png"])

st.sidebar.header("IS Code Settings")
exposure_class = st.sidebar.selectbox(
    "Exposure class (IS 456)",
    options=list(IS_456_CRACK_LIMITS_MM.keys()),
    index=0,
)
st.sidebar.caption(
    "IS 456:2000 crack-width check is shown against the selected exposure class."
)

is_table_df = pd.DataFrame(
    {
        "Exposure class": list(IS_456_CRACK_LIMITS_MM.keys()),
        "Max crack width as per IS 456 (mm)": list(IS_456_CRACK_LIMITS_MM.values()),
    }
)

with st.expander("IS 456 crack-width reference table", expanded=True):
    st.markdown("**Reference limits used in this app**")
    st.dataframe(is_table_df, use_container_width=True, hide_index=True)
    st.caption(
        "Guide used here: 0.30 mm for Mild, 0.20 mm for Moderate, and 0.10 mm for Severe / Very Severe / Extreme exposure."
    )

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption="Input Structural Image", use_container_width=True)

    binary = segment_crack(image)
    crack_pixels = int(binary.sum())

    st.subheader("Structural Crack Presence Analysis")
    if crack_pixels < 80:
        st.error("No structural crack detected.")
        st.stop()
    st.success("Crack detected.")

    analysis_mode = st.radio(
        "Measurement mode",
        ["Analyze all cracks", "Analyze longest crack only"],
        horizontal=True,
    )

    st.subheader("Calibration Step")
    st.write("Click two points across the 25 mm calibration sticker diameter.")

    coords = streamlit_image_coordinates(rgb, key="calibration_image")
    if coords is not None:
        click = (int(coords["x"]), int(coords["y"]))
        if click != st.session_state.last_click:
            if len(st.session_state.points) < 2:
                st.session_state.points.append(click)
            st.session_state.last_click = click

    if st.session_state.points:
        st.write(f"Selected points: {st.session_state.points}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset Calibration"):
            st.session_state.points = []
            st.session_state.last_click = None
            st.rerun()
    with col2:
        st.caption("Exactly two clicks are used for scale calibration.")

    if len(st.session_state.points) >= 2:
        (x1, y1), (x2, y2) = st.session_state.points[:2]
        pixel_d = math.hypot(x2 - x1, y2 - y1)

        if pixel_d < 2:
            st.error("Calibration points are too close. Please reset and click again.")
            st.stop()

        mm_per_pixel = 25.0 / pixel_d

        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))
        r = max(1, int(round(pixel_d / 2)))
        mask = np.ones(binary.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r + 8, 0, -1)
        binary = binary * mask

        if int(binary.sum()) == 0:
            st.error("After masking the calibration sticker, no crack pixels remain.")
            st.stop()

        components = connected_crack_components(binary)
        if not components:
            st.error("No valid crack components remained after calibration masking.")
            st.stop()

        if analysis_mode == "Analyze longest crack only":
            components = components[:1]

        st.success(f"Calibration complete. Scale = {mm_per_pixel:.4f} mm/pixel")
        st.info(f"Measured crack count: {len(components)}")
        st.info(f"IS 456 exposure class selected: {exposure_class}")

        results = []
        combined_overlay = image.copy()
        combined_mask = np.zeros_like(binary, dtype=np.uint8)

        for idx, comp in enumerate(components, start=1):
            try:
                metrics = measure_crack(comp["mask"], mm_per_pixel, exposure_class=exposure_class)
            except ValueError:
                continue

            results.append(
                {
                    "Crack": idx,
                    "Length (mm)": round(metrics["length_mm"], 2),
                    "Min width (mm)": round(metrics["min_width_mm"], 3),
                    "Avg width (mm)": round(metrics["avg_width_mm"], 3),
                    "Max width (mm)": round(metrics["max_width_mm"], 3),
                    "General type": metrics["general_classification"],
                    "IS 456 limit (mm)": round(metrics["is456_limit_mm"], 3),
                    "IS 456 status": metrics["is456_status"],
                    "widths_mm": metrics["widths_mm"],
                    "skeleton": metrics["skeleton"],
                    "mask": comp["mask"],
                    "is456_within_limit": metrics["is456_within_limit"],
                }
            )
            combined_mask = np.maximum(combined_mask, comp["mask"].astype(np.uint8))

            ys, xs = np.where(metrics["skeleton"])
            for x, y in zip(xs, ys):
                cv2.circle(combined_overlay, (int(x), int(y)), 1, (0, 0, 255), -1)

        if not results:
            st.error("Could not compute valid measurements from the detected cracks.")
            st.stop()

        st.subheader("Measured Crack Summary")
        summary_df = pd.DataFrame(
            [
                {k: v for k, v in row.items() if k not in {"widths_mm", "skeleton", "mask", "is456_within_limit"}}
                for row in results
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        longest = max(results, key=lambda x: x["Length (mm)"])
        widest = max(results, key=lambda x: x["Max width (mm)"])
        failures = sum(not r["is456_within_limit"] for r in results)

        c1, c2, c3 = st.columns(3)
        c1.metric("Longest measured crack", f"#{int(longest['Crack'])}", f"{longest['Length (mm)']:.2f} mm")
        c2.metric("Widest measured crack", f"#{int(widest['Crack'])}", f"{widest['Max width (mm)']:.3f} mm")
        c3.metric("Cracks exceeding IS 456 limit", f"{failures}")

        selected_crack_id = st.selectbox(
            "Show width-variation plot for",
            options=[int(r["Crack"]) for r in results],
            index=0,
        )
        selected = next(r for r in results if int(r["Crack"]) == int(selected_crack_id))

        st.subheader("Selected Crack Width Statistics")
        st.write(f"Length: {selected['Length (mm)']:.2f} mm")
        st.write(f"Minimum Width: {selected['Min width (mm)']:.3f} mm")
        st.write(f"Average Width: {selected['Avg width (mm)']:.3f} mm")
        st.write(f"Maximum Width: {selected['Max width (mm)']:.3f} mm")
        st.write(f"General Type: {selected['General type']}")
        st.write(f"IS 456 allowable limit for {exposure_class}: {selected['IS 456 limit (mm)']:.3f} mm")

        if selected["is456_within_limit"]:
            st.success(selected["IS 456 status"])
        else:
            st.error(selected["IS 456 status"])

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(selected["widths_mm"])
        ax.axhline(selected["IS 456 limit (mm)"], linestyle="--")
        ax.set_title(f"Crack #{selected_crack_id} Width Variation")
        ax.set_xlabel("Skeleton Position")
        ax.set_ylabel("Width (mm)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Crack Skeleton Overlay")
        st.image(cv2.cvtColor(combined_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("Critical Crack Regions")
        heat = crack_density_map(combined_mask)
        st.image(heat, use_container_width=True)
