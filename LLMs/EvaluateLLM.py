import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance

def compute_metrics(ground_truth, llm_output):
    keys = list(set(ground_truth.keys()).union(set(llm_output.keys())))
    
    y_true = [ground_truth.get(key, '') for key in keys]
    y_pred = [llm_output.get(key, '') for key in keys]

    # Convert to string for precision, recall, and F1 score
    y_true_str = [str(y) for y in y_true]
    y_pred_str = [str(y) for y in y_pred]

    precision = precision_score(y_true_str, y_pred_str, average='weighted', zero_division=1)
    recall = recall_score(y_true_str, y_pred_str, average='weighted', zero_division=1)
    f1 = f1_score(y_true_str, y_pred_str, average='weighted', zero_division=1)

    levenshtein_distances = [levenshtein_distance(gt, pred) / max(len(gt), len(pred), 1) for gt, pred in zip(y_true_str, y_pred_str)]
    avg_levenshtein_distance = np.mean(levenshtein_distances)

    vectorizer = TfidfVectorizer().fit(y_true_str + y_pred_str)
    y_true_tfidf = vectorizer.transform(y_true_str)
    y_pred_tfidf = vectorizer.transform(y_pred_str)
    cosine_similarities = [cosine_similarity(y_true_tfidf[i], y_pred_tfidf[i])[0, 0] for i in range(y_true_tfidf.shape[0])]
    avg_cosine_similarity = np.mean(cosine_similarities)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_levenshtein_distance": avg_levenshtein_distance,
        "avg_cosine_similarity": avg_cosine_similarity
    }

# Load JSON files
with open('../Results/Ground truth.json', 'r') as f:
    ground_truth_data = json.load(f)

#Change file name accordingly
with open('../Results/Llama3.json', 'r') as f:
    llm_output_data = json.load(f)

# Ensure the data is a list of records
ground_truth_records = ground_truth_data["Patients"]
llm_output_records = llm_output_data["Patients"]

# Calculate metrics for each record
all_metrics = []
for ground_truth, llm_output in zip(ground_truth_records, llm_output_records):
    metrics = compute_metrics(ground_truth, llm_output)
    # print(metrics)
    all_metrics.append(metrics)

# Calculate average metrics
avg_metrics = {
    "precision": np.mean([m["precision"] for m in all_metrics]),
    "recall": np.mean([m["recall"] for m in all_metrics]),
    "f1_score": np.mean([m["f1_score"] for m in all_metrics]),
    "avg_levenshtein_distance": np.mean([m["avg_levenshtein_distance"] for m in all_metrics]),
    "avg_cosine_similarity": np.mean([m["avg_cosine_similarity"] for m in all_metrics]),
}
rounded_avg_metrics = {k: round(v, 3) for k, v in avg_metrics.items()}


# Find max and min values
max_precision = round(max([m["precision"] for m in all_metrics]), 3)
max_recall = round(max([m["recall"] for m in all_metrics]), 3)
max_f1_score = round(max([m["f1_score"] for m in all_metrics]), 3)
max_cosine_similarity = round(max([m["avg_cosine_similarity"] for m in all_metrics]), 3)
min_levenshtein_distance = round(min([m["avg_levenshtein_distance"] for m in all_metrics]), 3)

# Print results
print("Average metrics:", rounded_avg_metrics)
print("Max precision:", max_precision)
print("Max recall:", max_recall)
print("Max F1 score:", max_f1_score)
print("Max cosine similarity:", max_cosine_similarity)
print("Min Levenshtein distance:", min_levenshtein_distance)