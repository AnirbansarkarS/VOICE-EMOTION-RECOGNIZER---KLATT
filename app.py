import streamlit as st
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import soundfile as sf
from utils.audio_features import extract_features
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os

# Load trained model
model = joblib.load("models/emotion_model.pkl")
emotion_labels = model.classes_

# Function: Record live audio
def record_audio(duration=5, fs=44100):
    st.info("🎙️ Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_audio.name, fs, recording)
    st.success("✅ Recording complete!")
    return temp_audio.name

# Function: Predict emotion
def predict_emotion(file_path):
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    return prediction, probs

# Function: Show emotion pie chart
def show_emotion_pie(probs, labels):
    fig, ax = plt.subplots()
    ax.pie(probs, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

# -------------- Streamlit UI --------------------

st.set_page_config(page_title="Voice Emotion Recognizer", page_icon="🎤")
st.title("🎧 Voice Emotion Recognition App")
st.markdown("Detect emotions from **live voice** or **uploaded audio files** using machine learning.")

st.divider()

# ----------- Live Recording Section -----------
st.subheader("🎙️ Record Live Voice")

if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = None

if st.button("🎤 Start Recording"):
    st.session_state.audio_file_path = record_audio(duration=5)
    st.audio(st.session_state.audio_file_path)

if st.session_state.audio_file_path:
    if st.button("🔍 Get Emotion from Recording"):
        predicted_emotion, probabilities = predict_emotion(st.session_state.audio_file_path)
        st.success(f"🧠 Predicted Emotion: **{predicted_emotion.upper()}**")
        show_emotion_pie(probabilities, emotion_labels)

st.divider()

# ----------- Upload File Section -----------
st.subheader("📁 Upload .WAV File")

if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None

uploaded_file = st.file_uploader("Upload your `.wav` file here", type=["wav"])
if uploaded_file:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.uploaded_file_path = "temp.wav"
    st.audio("temp.wav")

if st.session_state.uploaded_file_path:
    if st.button("📤 Get Emotion from File"):
        predicted_emotion, probabilities = predict_emotion(st.session_state.uploaded_file_path)
        st.success(f"🧠 Predicted Emotion: **{predicted_emotion.upper()}**")
        show_emotion_pie(probabilities, emotion_labels)

st.divider()
st.markdown("🛠️ Built with Python, Streamlit, and Machine Learning | Project by Anirban Sarkar 🚀")
