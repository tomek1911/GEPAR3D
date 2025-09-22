import json
import numpy as np

def compute_inverse_frequency_weights(json_file, output_file, exclude_background=True):
    # Load class occurrences
    with open(json_file, "r") as f:
        class_occurrences = json.load(f)

    # Convert keys back to int (JSON keys are strings)
    class_occurrences = {int(k): v for k, v in class_occurrences.items()}

    # Optionally exclude background (class 0)
    if exclude_background and 0 in class_occurrences:
        del class_occurrences[0]

    # Total number of samples (max occurrence count)
    total_samples = max(class_occurrences.values())

    # Compute weights as inverse frequency
    weights = {}
    for cls, count in class_occurrences.items():
        if count > 0:
            weights[cls] = total_samples / count
        else:
            weights[cls] = 0.0

    # Normalize weights so they sum to 1
    weight_array = np.zeros(max(class_occurrences.keys()) + 1, dtype=np.float32)
    for cls, w in weights.items():
        weight_array[cls] = w
    #normalize - sum to 1
    # weight_array /= weight_array.sum()
    
    # normalize so the mean of used classes is 1.0
    used = weight_array > 0
    if used.any():
        weight_array[used] *= (used.sum() / weight_array[used].sum())

    # Save as .npy
    np.save(output_file, weight_array)
    print(f"Saved inverse frequency weights to {output_file}")
    print("Weights:", weight_array)


if __name__ == "__main__":
    compute_inverse_frequency_weights(
        json_file="data/class_distribution.json",
        output_file="data/class_invfreq_weights.npy",
        exclude_background=False
    )
