import json
import numpy as np
import torch
import os

POI_CATEGORIES = ['aeroway', 'amenity', 'building', 'highway', 'landuse', 'leisure', 'natural', 'public_transport',
                  'railway', 'tourism', 'waterway']
NUM_POI_CATEGORIES = len(POI_CATEGORIES)

AUDIO_LABELS = [
    "Speech", "Children playing", "Laughter", "Shout or scream", "Footsteps", "Instrumental music",
    "Singing", "Amplified music", "Bird sounds", "Insects", "Crickets", "Dog", "Cat", "Poultry", "Horse",
    "Cows mooing", "Car", "Truck", "Bus", "Train", "Tram", "Metro", "Plane", "Helicopter", "Boat", "Bell",
    "Siren", "Alarm", "Vehicle horn", "Reverse beeper", "Explosion", "Gunshot", "Jackhammer", "Drill",
    "Crane/Bulldozer", "Flowing water", "Falling water", "Waves"
]
NUM_AUDIO_LABELS = len(AUDIO_LABELS)
AUDIO_LABEL_TO_IDX = {label: idx for idx, label in enumerate(AUDIO_LABELS)}


def calculate_and_save_priors(json_path, output_dir):
    """
    Calculates P(Event_Class | POI_Category) and P(Event_Class)
    from the provided JSON data and saves them.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {json_path}. Details: {e}")
        return
    except UnicodeDecodeError as e:
        print(
            f"Error: Unicode decoding error when reading {json_path}. Details: {e}. Ensure the file is UTF-8 encoded.")
        return


    event_poi_cooccurrence_counts = np.zeros((NUM_POI_CATEGORIES, NUM_AUDIO_LABELS), dtype=float)
    poi_category_occurrence_counts = np.zeros(NUM_POI_CATEGORIES, dtype=float)
    event_occurrence_counts = np.zeros(NUM_AUDIO_LABELS, dtype=float)

    valid_entries = 0

    for entry in data:
        poi_multi_hot = entry.get("poi_multi_hot")
        segments = entry.get("segments", [])

        if poi_multi_hot is None or len(poi_multi_hot) != NUM_POI_CATEGORIES:
            continue

        valid_entries += 1

        present_event_indices = set()
        for segment in segments:
            label = segment.get("label")
            if label in AUDIO_LABEL_TO_IDX:
                present_event_indices.add(AUDIO_LABEL_TO_IDX[label])

        for event_idx in present_event_indices:
            event_occurrence_counts[event_idx] += 1

        for poi_cat_idx in range(NUM_POI_CATEGORIES):
            if poi_multi_hot[poi_cat_idx] == 1:
                poi_category_occurrence_counts[poi_cat_idx] += 1
                for event_idx in present_event_indices:
                    event_poi_cooccurrence_counts[poi_cat_idx, event_idx] += 1

    if valid_entries == 0:
        print("Error: No valid entries found in the JSON data to calculate priors.")
        return

    alpha = 1.0

    numerator_cond = event_poi_cooccurrence_counts + alpha
    denominator_cond = poi_category_occurrence_counts[:, np.newaxis] + alpha * NUM_AUDIO_LABELS

    conditional_priors_poi_event = numerator_cond / denominator_cond

    numerator_marg = event_occurrence_counts + alpha
    denominator_marg = valid_entries + alpha * NUM_AUDIO_LABELS
    marginal_event_priors = numerator_marg / denominator_marg

    os.makedirs(output_dir, exist_ok=True)
    torch.save(torch.from_numpy(conditional_priors_poi_event).float(),
               os.path.join(output_dir, "statistical_poi_event_priors.pt"))
    torch.save(torch.from_numpy(marginal_event_priors).float(),
               os.path.join(output_dir, "statistical_marginal_event_priors.pt"))

    print(f"Statistical priors calculated from {valid_entries} valid entries and saved to {output_dir}")
    print(f"  Shape of P(Event|POI_Category): {conditional_priors_poi_event.shape}")
    print(f"  Shape of P(Event): {marginal_event_priors.shape}")
    print(f"Example P(Event|{POI_CATEGORIES[0]}): {conditional_priors_poi_event[0, :5]}...")
    print(f"Example P(Event): {marginal_event_priors[:5]}...")


if __name__ == "__main__":
    json_file_path = os.path.join("..", "outputs", "1_poi_features.json")

    priors_output_dir = os.path.join("..", "outputs", "statistical_priors")

    calculate_and_save_priors(json_file_path, priors_output_dir)

