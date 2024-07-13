import streamlit as st
from scipy.io import wavfile
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


# Load the pre-trained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Function to transcribe audio
def transcribe_audio(audio_file):
    input_audio, _ = librosa.load(audio_file, sr=16000)
    input_values = tokenizer(input_audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription

st.title("Audio to Text Transcription")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])



if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the audio file
    st.audio("temp_uploaded_audio.wav")

    # Transcribe the uploaded audio file
    st.write("Transcribing uploaded audio...")
    transcription = transcribe_audio("temp_uploaded_audio.wav")
    st.write("Transcription:", transcription)


