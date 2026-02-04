import streamlit as st
import numpy as np

from model.model_loader import load_model
from utils.config import HF_REPO_ID,HF_MODEL_FILENAME,CLASSES
from audio.preprocessing import load_audio,extract_mfcc
from ui.visualizations import plot_waveform
from utils.logger import setup_logger

logger=setup_logger("StreamlitApp")


#load model
model=load_model()

st.title("Heart Murmur Detection using LSTM")
uploaded_file=st.file_uploader(
    "Upload a heart sound(WAV/MP3)",
    type=["wav","mp3"]

)

if uploaded_file is not None:
    try:
        y,sr=load_audio(uploaded_file)
        st.subheader("Waveform of Input Sound")
        fig=plot_waveform(y,sr)
        st.pyplot(fig)

        X_input=extract_mfcc(y,sr)

        prediction=model.predict(X_input)
        prediction_class=np.argmax(prediction,axis=1)[0]
        prediction_label=CLASSES.get(prediction_class, "Unknown")

        st.subheader("prediction result")
        st.write(f"Predicted class: **{prediction_label}**")
        st.write("Raw Prediction Scores:",prediction)

        logger.info("Prediction completed successfully")

    except Exception as e:
        logger.exception("Inference pipeline failed")
        st.error(f"An error occurred while processing the audio file: {e}")