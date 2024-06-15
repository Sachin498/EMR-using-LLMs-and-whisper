import streamlit as st
import whisper
import os
import tempfile
import requests
import json
import torch

# Load the Whisper model and move it to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base")

# OLLAMA endpoint and configuration
OLLAMA_ENDPOINT = "https://0ad5-2a02-a210-22c0-aa00-4d8d-99f5-faa7-79e2.ngrok-free.app/api/generate"
OLLAMA_MODEL = "mistral:instruct"
SYSTEM_PROMPT = "Generate an EMR from the given conversation."

# Streamlit app
st.title("Audio Transcription and EMR Generation App")
st.write("Upload an audio file, and it will be transcribed to text using OpenAI's Whisper. Then, the transcribed text will be used to generate an EMR using the OLLAMA API.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Verify the temporary file path
    if os.path.exists(temp_file_path):
        # Button to transcribe the audio file and generate EMR
        if st.button("Transcribe and Generate EMR"):
            try:
                # Transcribe the audio file
                st.write("Transcribing audio...")

                # Load and preprocess the audio file
                audio = whisper.load_audio(temp_file_path)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(device)

                # Move model to device and perform transcription
                model = model.to(device)
                options = whisper.DecodingOptions(task="translate")
                result = whisper.decode(model, mel, options)
                conversation_string = result.text

                # Display the transcribed text
                st.write("Transcribed Text:")
                st.write(conversation_string)

                # Prepare the prompt and data for OLLAMA API
                OLLAMA_PROMPT = f"{conversation_string}:\n{SYSTEM_PROMPT}"
                OLLAMA_DATA = {
                    "model": OLLAMA_MODEL,
                    "prompt": OLLAMA_PROMPT,
                    "format": "json",
                    "stream": False,
                    "keep_alive": "5m",
                }
                headers = {
                    "Content-Type": "application/json"
                }

                # Send the request to the OLLAMA API
                st.write("Generating EMR...")
                try:
                    response = requests.post(OLLAMA_ENDPOINT, headers=headers, data=json.dumps(OLLAMA_DATA))
                    response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
                    if response.status_code == 200:
                        emr = response.json().get("response", "No text returned from the API.")
                        st.write("Generated EMR:")
                        st.write(emr)
                    else:
                        st.write("Error in generating EMR. Status Code:", response.status_code)
                        st.write(response.json())
                except requests.exceptions.RequestException as e:
                    st.write("An error occurred while connecting to the OLLAMA API:")
                    st.write(str(e))

            except Exception as e:
                st.write("An error occurred during transcription or EMR generation:")
                st.write(str(e))

        # Clean up the temporary file
        os.remove(temp_file_path)
    else:
        st.write("Temporary file not found. Please try uploading the file again.")
