import streamlit as st
import whisper
import os
import tempfile
import requests
import json
import torch
from streamlit_mic_recorder import mic_recorder
import base64
import gc

torch.cuda.empty_cache()
gc.collect()
# Path to your background image
background_image_path = './Images/BG9.jpg'

# Read the image file and encode it as a base64 string
with open(background_image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Inject CSS to add the background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpeg;base64,{encoded_string});
        background-position: center;
        background-size: cover;

    }}
    </style>
    """,
    unsafe_allow_html=True
)


# OLLAMA endpoint and configuration
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"  # Update with your actual endpoint
SYSTEM_PROMPT = "Identify and extract the entities mentioned below and generate an EMR from the above conversation in the specified format."


# Center the title
background_color = "#B8C4C2"
st.markdown("<h1 style='text-align: center; '>EMR Generation</font></h1>", unsafe_allow_html=True)
st.write("Upload an audio file or record your voice, and it will be transcribed to text using OpenAI's Whisper. Then, the transcribed text will be used to generate an EMR using the OLLAMA API.")

# Sidebar for field selection and output format
st.sidebar.title("EMR Configuration")
st.sidebar.write("Select the fields you want to include in the EMR:")

st.session_state.setdefault('transcribed_text', "")
st.session_state.setdefault('ollama_prompt', "")
st.session_state.setdefault('custom_fields', [])
st.session_state.setdefault('fields', [])


# Field selection including custom fields
all_fields = ["Name", "Age", "Gender", "Diagnosis", "Symptoms", "Medications", "Allergies", "Notes"] 
selected_fields = st.sidebar.multiselect(
    "Fields",
    all_fields
)

# Update session state with selected fields
st.session_state['fields'] = selected_fields

# Custom field input
custom_field = st.sidebar.text_input("Add a custom field:")
add_custom_field = st.sidebar.button("Add")

if add_custom_field and custom_field:
    if custom_field not in st.session_state['custom_fields']:
        st.session_state['custom_fields'].append(custom_field)
    custom_field = ""

# Display checkboxes for custom fields, checked by default
st.sidebar.write("Custom Fields:")
for field in st.session_state['custom_fields']:
    st.sidebar.checkbox(field, key=f"custom_field_{field}", value=True)

output_format = st.sidebar.selectbox("Select the output format for the EMR:", ["JSON", "Text"], index=1)

# Developer option toggle
developer_option = st.sidebar.toggle("Developer option")

# LLM selection
llm_options = ["Llama2 basic", "Llama3 basic", "mistral basic", "Llama2 Opt", "LLama3 Opt"]
default_llm = "LLama3 Opt"
if 'llm_selection' not in st.session_state:
    st.session_state['llm_selection'] = default_llm

if developer_option:
    st.session_state['llm_selection'] = st.sidebar.selectbox("Select the LLM configuration:", llm_options)
else:
    st.sidebar.write(f"LLM Configuration: {st.session_state['llm_selection']}")

# Define the LLM configurations
llm_configs = {
    "LLama3 Opt": {"model": "llama3", "options": {"temperature": 0.5, "num_ctx": 4096, "mirostat": 1}},
    "Llama2 Opt": {"model": "llama2", "options": {"temperature": 0.5, "num_ctx": 4096, "top_k": 20, "top_p": 0.8}},
    "Llama2 basic": {"model": "llama2"},
    "Llama3 basic": {"model": "llama3"},
    "mistral basic": {"model": "mistral"}, 
}

selected_llm_config = llm_configs[st.session_state['llm_selection']]

# File uploader
st.session_state['audio_data'] = st.file_uploader("Choose an audio file or Record an audio", type=["mp3", "wav", "m4a"])
uploaded_file = st.session_state['audio_data']


device = "cuda" if torch.cuda.is_available() else "cpu"
audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    format="wav",
    callback=None,
    args=(),
    kwargs={},
    key=None
)

if uploaded_file is not None or audio:
    # Save the uploaded file or recorded audio to a temporary file
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
    else:
        # Save the recorded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
            with open(temp_file_path, 'wb') as f:
                f.write(audio['bytes'])
                st.write("Audio recorded successfully")
        
        # recording_message.empty()
        st.write("Audio recorded and saved as", temp_file_path)
        


    # Verify the temporary file path
    if os.path.exists(temp_file_path):
        
        
        # Button to transcribe the audio file and generate EMR
        if st.button("Transcribe"):
            try:
                model = whisper.load_model("medium")
                # Transcribe the audio file
                transcribe_message = st.empty()
                transcribe_message.write("Transcribing audio...")

                # Load and preprocess the audio file
                audio = whisper.load_audio(temp_file_path)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(device)

                # Move model to device and perform transcription
                model = model.to(device)
                options = whisper.DecodingOptions(task="translate")
                result = whisper.decode(model, mel, options)
                conversation_string = result.text

                # Store the transcribed text in session state
                st.session_state['transcribed_text'] = conversation_string

                # Update the transcribed text area
                transcribe_message.empty()

                
            except Exception as e:
                st.write("An error occurred during transcription:")
                st.write(str(e))

        # Display the transcribed text if it exists
        if st.session_state['transcribed_text']:
            st.text_area("Transcribed Text:", st.session_state['transcribed_text'], height=200)
            # Prepare the prompt and data for OLLAMA API
            fields_prompt = "\n".join(st.session_state['fields'] + st.session_state['custom_fields'])

            default_prompt = f"{st.session_state['transcribed_text']}\n\n{SYSTEM_PROMPT}\nInclude the following fields in the format specified below:\n{fields_prompt}\nOnly output the EMR in plain text without any additional text. Do not add \"Here is the extracted information:\" and \"Let me know if you'd like help with anything else\""
            ollama_prompt = default_prompt

            if developer_option:
                ollama_prompt = st.text_area("Edit the prompt before generating EMR:", value=default_prompt, height=200)

            # Update the session state with the edited or default prompt
            st.session_state['ollama_prompt'] = ollama_prompt

            if st.button("Generate EMR"):
                OLLAMA_DATA = {
                    "model": selected_llm_config["model"],
                    "prompt": st.session_state['ollama_prompt'],
                    "stream": False,
                    "keep_alive": "5m",
                }

                # Add options if present in the selected LLM config
                if "options" in selected_llm_config:
                    OLLAMA_DATA["options"] = selected_llm_config["options"]

                headers = {
                    "Content-Type": "application/json"
                }

                # Send the request to the OLLAMA API
                generating_message = st.empty()  # Create a placeholder
                generating_message.write("Generating EMR...")
                try:
                    response = requests.post(OLLAMA_ENDPOINT, headers=headers, data=json.dumps(OLLAMA_DATA))
                    response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
                    if response.status_code == 200:
                        emr = response.json().get("response", "No text returned from the API.")
                        generating_message.empty()  # Clear the "Generating EMR..." message
                        # st.write("Generated EMR:")
                        st.text_area("Generated EMR:", emr, height=300)
                        # Display and download the EMR in the selected format
                        if output_format == "JSON":
                            # st.json(emr)
                            st.download_button(
                                label="Download EMR",
                                data=json.dumps(emr, indent=4),
                                file_name="EMR.json",
                                mime="application/json",
                            )
                        else:
                            # st.text(emr)
                            st.download_button(
                                label="Download EMR",
                                data=emr,
                                file_name="EMR.txt",
                                mime="text/plain",
                            )
                    else:
                        st.write("Error in generating EMR. Status Code:", response.status_code)
                        st.write(response.json())
                except requests.exceptions.RequestException as e:
                    st.write("An error occurred while connecting to the OLLAMA API:")
                    st.write(str(e))

        # Clean up the temporary file
        os.remove(temp_file_path)
    else:
        st.write("Temporary file not found. Please try uploading the file again.")