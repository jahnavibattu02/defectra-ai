from __future__ import annotations

import base64
import os
import requests
import streamlit as st
from PIL import Image
from io import BytesIO

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI Vision | Bottle Inspection", page_icon="🔎", layout="wide")
st.title("AI Vision")
st.caption("MVTec AD bottle anomaly detection · CNN Autoencoder vs ResNet18 feature distance")

with st.sidebar:
    st.header("Inspection settings")
    selected_model = st.selectbox(
        "Model",
        ["Both models", "CNN Autoencoder", "ResNet18 feature-based"],
        help="The two models have separate anomaly-score scales and thresholds.",
    )
    model_value = {
        "Both models": "both",
        "CNN Autoencoder": "autoencoder",
        "ResNet18 feature-based": "resnet",
    }[selected_model]
    st.info("Use clear, well-lit bottle images. This demo is not a certified industrial inspection system.")

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if not health.get("models_loaded"):
        st.error("The API is reachable, but model artifacts are not ready. Check the API terminal and README export instructions.")
        if health.get("detail"):
            st.code(health["detail"])
        st.stop()
except requests.RequestException:
    st.error(f"Cannot reach the API at {API_URL}. Start the FastAPI server first.")
    st.stop()

uploaded = st.file_uploader("Upload a bottle image", type=["png", "jpg", "jpeg", "webp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Input image")
        st.image(image, use_container_width=True)
        st.caption(f"Image size: {image.width} × {image.height}")
    if st.button("Inspect bottle", type="primary", use_container_width=True):
        with st.spinner("Running selected model(s)..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/png")},
                    data={"model": model_value},
                    timeout=120,
                )
                if response.status_code != 200:
                    st.error(f"API error ({response.status_code})")
                    st.code(response.text)
                    st.stop()
                payload = response.json()
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

        st.subheader("Inspection results")
        result_items = list(payload["results"].items())
        cols = st.columns(len(result_items))
        for col, (key, result) in zip(cols, result_items):
            with col:
                st.markdown(f"### {result['model'].replace('_', ' ').title()}")
                if result["prediction"] == "normal":
                    st.success("Prediction: NORMAL")
                else:
                    st.warning("Prediction: POTENTIAL DEFECT")
                st.metric("Anomaly score", f"{result['anomaly_score']:.6f}")
                st.caption(f"Validation threshold: {result['threshold']:.6f}")
                heatmap_bytes = base64.b64decode(result["heatmap_png_base64"])
                st.image(Image.open(BytesIO(heatmap_bytes)), caption="Normalized anomaly heatmap", use_container_width=True)
                st.download_button(
                    "Download heatmap PNG",
                    data=heatmap_bytes,
                    file_name=f"{key}_heatmap.png",
                    mime="image/png",
                    key=f"download_{key}",
                )
        st.caption(payload["notice"])
        report = {
            "image_name": uploaded.name,
            "image_width": payload["image_width"],
            "image_height": payload["image_height"],
            "results": {
                key: {k: v for k, v in result.items() if k != "heatmap_png_base64"}
                for key, result in payload["results"].items()
            },
            "notice": payload["notice"],
        }
        import json
        st.download_button(
            "Download JSON inspection report",
            data=json.dumps(report, indent=2),
            file_name="inspection_report.json",
            mime="application/json",
        )
else:
    st.info("Upload a bottle image to start an inspection.")
