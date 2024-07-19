# LLMs

This repository contains files and instructions for running and evaluating individual Large Language Models (LLMs) in generating Electronic Medical Records (EMRs) and comparing them against a manually annotated ground truth.

## Dataset and Ground Truth

The ground truth data is manually annotated by the author and is based on 50 transcripts found in the [transcripts folder](../Audio Transcripts).
he entire dataset for these transcripts can be accessed at [Springer Nature's Figshare collection](https://springernature.figshare.com/collections/A_dataset_of_simulated_patient-physician_medical_interviews_with_a_focus_on_respiratory_cases/5545842/1)

## Prerequisites
-  Ollama Installation: Ensure that Ollama is downloaded and installed. Additionally, install the models you intend to use.
-  Ollama Runtime: Make sure Ollama is running.

## Prompts Generation
The prompts are generated using the script located at [file](PromptGen.py). This script retrieves the necessary fields from the ground truth data and creates prompts for each file used.

## Model Testing
To test the models, use the ["TestModel.py"](TestModel.py) script. TThis script fetches prompts from the ["prompts.json"](prompts.json) file, makes requests to the running model, and saves the responses in the specified directory.

## Evaluation
The ["Evaluate.py"](EvaluateLLM.py) script is used to evaluate the performance of the LLMs against the ground truth. Before running the evaluation, ensure that the file paths for the LLM's responses are correctly specified in the script. The evaluation includes metrics such as precision, recall, F1 score, Levenshtein distance, and cosine similarity.
