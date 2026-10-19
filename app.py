import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fake_news_model.joblib"

st.title("📰 Fake News Detection")
st.write("Check a news headline or article using a machine learning model.")

if not MODEL_PATH.exists():
    st.error("Model not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)

news_text = st.text_area(
    "Enter news text:",
    placeholder="Paste a news headline or article here...",
    height=180
)

if st.button("Check News"):
    if not news_text.strip():
        st.warning("Please enter some news text.")
    else:
        prediction = model.predict([news_text])[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([news_text])[0]
            confidence = max(probabilities) * 100
        else:
            confidence = None

        label = str(prediction).upper()

        if label == "REAL":
            st.success("Prediction: REAL")
        elif label == "FAKE":
            st.error("Prediction: FAKE")
        else:
            st.info(f"Prediction: {label}")

        if confidence is not None:
            st.write(f"Model confidence: {confidence:.1f}%")

        st.caption(
            "This is a model prediction, not proof that a claim is true or false. "
            "Verify news using reliable sources."
        )
