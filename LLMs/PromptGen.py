import json

def remove_values(data):
    if isinstance(data, dict):
        return {k: remove_values(v) if isinstance(v, (dict, list)) else "" for k, v in data.items()}
    elif isinstance(data, list):
        return [remove_values(v) for v in data]
    else:
        return ""


# Load the data from the patient_records.json file
with open("../Results/Ground truth.json") as f:
    data = json.load(f)

# Initialize an empty list to store the prompts
prompts = []

for patient in data["Patients"]:
    patient_ID = patient["Patient ID"]
    output_data = remove_values(patient)
    del output_data["Patient ID"]

    system_prompt = "Generate a structured Electronic Medical Record (EMR) in JSON format from the provided doctor-patient conversation transcript. Ensure the EMR includes the following key-value pairs and organizes the information accordingly:\n" + json.dumps(output_data, indent=4) + "\nOnly output the EMR without any additional text. Output only what you find from the conversation. If something is not found add NA."
    prompt = {
        "Patient ID": patient_ID,
        "Prompt": system_prompt,
        "EMR": output_data
    }

    prompts.append(prompt)

    file_path = "prompts.json"

    # Save the prompts list to a JSON file
    with open(file_path, "w") as file:
        json.dump(prompts, file, indent=4)
