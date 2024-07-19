
# EMR Generating App using LLMs and whisper

This repository contains the code and resources for generating Electronic Medical Records (EMRs) from doctor-patient conversations using whisper and Lightweight Large Language Models (LLMs) namely llama2, llama3 and mistral. The application allows users to upload recordings or record live the conversations between doctors and patients during an appointment. Using whisper and the LLMs mentioned, the application generates an EMR which can be downloaded.



## Repository Structure

- Audio Transcripts

    Contains the transcripts of the audio recordings used for testing and evaluation.

- LLMs

    Contains the code to run individual models on Ollama to generate EMRs from the transcripts

- Results

    Contains the results from running individual models along with the manually annotated ground truth

- EMRGeneratingApp

    Streamlit application for generating EMRs from audio transcripts.

- EvaluateLLM.py

    Script for evaluating the performance of different LLM models.



## Getting Started


## Prerequisites

Download and install ollama from https://ollama.com/ 

Open the terminal and install the model you want to use. For example:

```bash
  ollama pull llama3
```
You can see the installed models using 
```bash
    ollama list
```
Ensure you have Python installed. Then, install the required packages using the following command:

```bash
  pip install -r requirements.txt
```

### Running the application
In a terminal tab, enter the following command

```bash
    ollama start 
```
or
```bash
    ollama serve
```

In another tab, while in the same directory as the Repository, enter the command:

```bash
    streamlit run EMRGeneratingApp.py 
```
## Instructions
Users can either upload recordings of doctor-patient conversations or record them live during the visit. The side panel includes a drop-down menu to select the fields they want to extract from the conversation, with an option to add custom fields as needed. Users can also choose the output format, currently supporting text and JSON formats.
### Steps:
-    Upload/Record Audio: Upload pre-recorded audio files or record the conversation live within the application.
-    Select Fields: Use the drop-down menu to choose the specific fields to extract from the conversation. Custom fields can be added as needed.
-    Choose Output Format: Select the desired output format (text or JSON).
-    Transcribe Audio: Click the "Transcribe" button. The Whisper model will generate transcripts in English from the audio, regardless of the original language.
-    Generate EMR: After transcription, click the "Generate EMR" button. The selected LLM will generate the EMR in the specified format, which can then be downloaded.

For advanced users, there is a developer option that allows the use of different LLMs and customization of the prompts given to the LLM.

## Additional Information

The model names and parameters can be updated in EMRGeneratingApp.py



