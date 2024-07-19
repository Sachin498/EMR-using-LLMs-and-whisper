import json
import requests

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

file_path = "prompts.json"

# Load the prompts from the JSON file
with open(file_path, "r") as file:
    prompts = json.load(file)
    for prompt in prompts:
        patient_ID = prompt["Patient ID"]
        system_prompt = prompt["Prompt"]
        file_path = f'..\\Audio Transcripts\\{patient_ID}.txt'
        with open(file_path, 'r') as file:
            conversation_string = file.read()

    
        OLLAMA_PROMPT = f"{conversation_string}:\n{system_prompt}"
        # Model name can be changed to specific models
        OLLAMA_DATA = {
            "model": "mistral:instruct",
            "prompt": OLLAMA_PROMPT,
            "format": "json",
            "stream": False,
            "keep_alive": "5m",
        }
        # print(OLLAMA_PROMPT)

        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(OLLAMA_ENDPOINT, headers=headers, data=json.dumps(OLLAMA_DATA))
        
        response_data = response.json()
        file_name = "responses.json"
        with open(file_name, "a") as file:
            json.dump(response_data, file, indent=4)
            file.write("\n")  # Add a newline after each JSON object
            print("ADDED")