
# EMR Generating App using LLMs and whisper

This repository contains the code and resources for generating Electronic Medical Records (EMRs) from doctor-patient conversations using whisper and Lightweight Large Language Models (LLMs) namely llama2, llama3 and mistral.



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


## Additional Information

The model names and parameters can be updated in EMRGeneratingApp.py

