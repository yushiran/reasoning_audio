import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
import json
import os
import pandas as pd
import torch.nn as nn
from CLAP.CLAPWrapper import CLAPWrapper  # Make sure this path is correct
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# --- End Definitions ---

def collate_fn(batch):
    audio_ids, file_paths, event_spans, labels, poi_multi_hots = zip(*batch)

    return (
        list(audio_ids),
        list(file_paths),
        list(event_spans),
        torch.stack(labels, 0),
        torch.stack(poi_multi_hots, 0)
    )


class AudioDataset(Dataset):
    def __init__(self, feature_dir, poi_json_file_path, transform=None, audio_duration=60,
                 sample_rate=16000):
        self.feature_dir = feature_dir
        self.all_data = []  # Will store items from poi_json_file_path
        self.audio_ids = []
        self.audio_id_to_poi_hot = {}
        self.audio_id_to_segments = {}  # To store audio spans/segments

        try:
            with open(poi_json_file_path, 'r', encoding='utf-8') as f:
                loaded_json_data = json.load(f)

            for item in loaded_json_data:
                audio_id = str(item.get('id'))
                if not audio_id:
                    print(f"Warning: Item found with no 'id'. Skipping: {item}")
                    continue

                self.all_data.append(item)  # Store the whole item
                self.audio_ids.append(audio_id)

                # Load POI multi-hot data
                if 'poi_multi_hot' in item and len(item['poi_multi_hot']) == NUM_POI_CATEGORIES:
                    self.audio_id_to_poi_hot[audio_id] = item['poi_multi_hot']
                else:
                    # print(f"Warning: POI multi-hot data missing or incorrect for id {audio_id}. Using zeros.")
                    self.audio_id_to_poi_hot[audio_id] = [0] * NUM_POI_CATEGORIES

                # Load audio segments (formerly audio_spans from labels.txt)
                # Assuming 'segments' in 1_poi_features.json has the same structure as 'audio_spans'
                self.audio_id_to_segments[audio_id] = item.get('segments', [])


        except FileNotFoundError:
            print(f"Error: POI JSON file not found at {poi_json_file_path}.")
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from {poi_json_file_path}. Details: {e}.")
        except UnicodeDecodeError as e:
            print(
                f"Error: Unicode decoding error when reading {poi_json_file_path}. Details: {e}. Ensure the file is UTF-8 encoded.")

        if not self.audio_ids:
            print("Warning: No audio IDs were loaded from the JSON file. The dataset will be empty.")

        self.label_map = AUDIO_LABEL_TO_IDX
        self.max_len = audio_duration * sample_rate
        self.transform = transform
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_ids)

    def __getitem__(self, idx):
        audio_id = self.audio_ids[idx]
        wav_path = os.path.join(self.feature_dir, f"{audio_id}.wav")

        labels = torch.zeros(NUM_AUDIO_LABELS, dtype=torch.float)

        # Get segments for the current audio_id
        current_segments = self.audio_id_to_segments.get(audio_id, [])

        event_spans_for_item = []
        for span in current_segments:
            label_text = span.get("label")
            if label_text and label_text in self.label_map:
                labels[self.label_map[label_text]] = 1.0
            event_spans_for_item.append({
                "start": span.get("start", 0.0),  # Default to 0.0 if missing
                "end": span.get("end", 0.0),  # Default to 0.0 if missing
                "label": label_text
            })

        # Get POI multi-hot vector
        poi_multi_hot_list = self.audio_id_to_poi_hot.get(audio_id, [0] * NUM_POI_CATEGORIES)
        poi_multi_hot = torch.tensor(poi_multi_hot_list, dtype=torch.float)

        return audio_id, wav_path, event_spans_for_item, labels, poi_multi_hot


class AudioClassifier(nn.Module):
    def __init__(self, num_labels, model_weights_path, device, use_poi=False, prior_strength=0.5, priors_dir=None):
        super().__init__()
        self.use_poi = use_poi
        self.device = device
        self.prior_strength = prior_strength
        self.num_labels = num_labels

        self.clap_wrapper = CLAPWrapper(model_fp=model_weights_path, device=device)
        for param in self.clap_wrapper.clap.parameters():
            param.requires_grad = False

        self.audio_event_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        ).to(device)

        if self.use_poi:
            if priors_dir is None:
                raise ValueError("priors_dir must be provided when use_poi is True")
            try:
                prior_file_path = os.path.join(priors_dir, "statistical_poi_event_priors.pt")
                marginal_prior_file_path = os.path.join(priors_dir, "statistical_marginal_event_priors.pt")

                self.statistical_poi_event_priors = torch.load(prior_file_path).to(device)
                self.statistical_marginal_event_priors = torch.load(marginal_prior_file_path).to(device)

                if self.statistical_poi_event_priors.shape != (NUM_POI_CATEGORIES, num_labels):
                    raise ValueError(
                        f"Loaded statistical_poi_event_priors has incorrect shape: {self.statistical_poi_event_priors.shape}. Expected: ({NUM_POI_CATEGORIES}, {num_labels})")
                if self.statistical_marginal_event_priors.shape != (num_labels,):
                    raise ValueError(
                        f"Loaded statistical_marginal_event_priors has incorrect shape: {self.statistical_marginal_event_priors.shape}. Expected: ({num_labels},)")
                print(f"Successfully loaded statistical priors for Bayesian model.")

            except FileNotFoundError:
                print(
                    f"Error: Prior files not found in {priors_dir}. Bayesian POI integration will not work correctly.")
                self.use_poi = False  # Fallback
            except Exception as e:
                print(f"Error loading priors: {e}. Bayesian POI integration will not work correctly.")
                self.use_poi = False  # Fallback

        self.time_predictor = nn.Linear(1024, 2).to(device)

    def forward(self, file_paths: list, batch_poi_multi_hot=None):
        audio_embeddings = self.clap_wrapper.get_audio_embeddings(file_paths, resample=True)
        if audio_embeddings.dim() == 3:
            audio_embeddings = audio_embeddings.mean(dim=1)

        audio_logits = self.audio_event_classifier(audio_embeddings.to(torch.float32))
        time_outputs = self.time_predictor(audio_embeddings.to(torch.float32))
        final_logits = audio_logits

        if self.use_poi and batch_poi_multi_hot is not None and hasattr(self, 'statistical_poi_event_priors'):
            batch_poi_multi_hot = batch_poi_multi_hot.to(self.device)
            prior_logits_list = []

            for i in range(batch_poi_multi_hot.size(0)):
                sample_poi_hot = batch_poi_multi_hot[i]
                active_poi_indices = sample_poi_hot.nonzero(as_tuple=True)[0]

                if len(active_poi_indices) == 0:  # No active POIs
                    sample_prior_probs = self.statistical_marginal_event_priors
                else:
                    active_poi_event_probs = self.statistical_poi_event_priors[active_poi_indices, :]
                    sample_prior_probs = active_poi_event_probs.mean(dim=0)

                sample_prior_logits = torch.log(sample_prior_probs + 1e-9)  # Add epsilon for numerical stability
                prior_logits_list.append(sample_prior_logits)

            if prior_logits_list:  # Ensure list is not empty
                prior_logits = torch.stack(prior_logits_list)
                final_logits = audio_logits + self.prior_strength * prior_logits

        return final_logits, time_outputs


def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    time_criterion = nn.MSELoss()

    for audio_ids, file_paths, event_spans, labels, poi_multi_hots in dataloader:
        labels = labels.to(model.device)
        poi_multi_hots = poi_multi_hots.to(model.device)

        time_labels_list = []
        for spans in event_spans:  # spans is a list of dicts for one audio file
            if spans:  # Check if the list of spans is not empty
                # Use the first span's start and end times
                time_labels_list.append([spans[0]['start'], spans[0]['end']])
            else:  # No spans, use default [0.0, 0.0]
                time_labels_list.append([0.0, 0.0])
        time_labels = torch.tensor(time_labels_list, dtype=torch.float32).to(device)

        outputs, time_outputs = model(file_paths, batch_poi_multi_hot=poi_multi_hots if model.use_poi else None)

        class_loss = criterion(outputs, labels)
        time_loss = time_criterion(time_outputs, time_labels)
        total_loss = class_loss + 0.3 * time_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

    return running_loss / len(dataloader)


def evaluate(model, dataloader):
    model.eval()
    all_preds_scores = []
    all_preds_binary = []
    all_labels = []
    all_time_preds = []
    all_time_labels = []

    with torch.no_grad():
        for audio_ids, file_paths, event_spans, labels, poi_multi_hots in dataloader:
            labels_device = labels.to(model.device)
            poi_multi_hots_device = poi_multi_hots.to(model.device)

            outputs, time_outputs = model(file_paths,
                                          batch_poi_multi_hot=poi_multi_hots_device if model.use_poi else None)

            scores = torch.sigmoid(outputs).cpu()
            preds_binary = (scores > 0.5).numpy()

            all_preds_scores.append(scores.numpy())
            all_preds_binary.append(preds_binary)
            all_labels.append(labels.cpu().numpy())

            all_time_preds.extend(time_outputs.cpu().numpy())

            batch_time_labels_list = []
            for spans in event_spans:
                if spans:
                    batch_time_labels_list.append([spans[0]['start'], spans[0]['end']])
                else:
                    batch_time_labels_list.append([0.0, 0.0])
            all_time_labels.extend(batch_time_labels_list)

    all_preds_scores = np.vstack(all_preds_scores)
    all_preds_binary = np.vstack(all_preds_binary)
    all_labels = np.vstack(all_labels)

    mAP = 0.0
    macro_f1 = 0.0
    accuracy = 0.0

    if np.sum(all_labels) > 0:  # Check if there are any positive labels at all
        mAP = average_precision_score(all_labels, all_preds_scores, average="macro")
        macro_f1 = f1_score(all_labels, all_preds_binary, average="macro", zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds_binary)

    IoU = compute_iou(all_time_preds, all_time_labels)
    return mAP, macro_f1, accuracy, IoU


def compute_iou(preds, labels_list):
    iou_scores = []
    preds_np = np.array(preds)
    labels_np = np.array(labels_list)

    for i in range(len(preds_np)):
        pred_start, pred_end = preds_np[i]
        label_start, label_end = labels_np[i]

        # Ensure pred_end is greater than pred_start for a valid prediction interval
        if pred_end <= pred_start:
            iou_scores.append(0.0)  # Invalid prediction interval
            continue

        # Handle cases where label interval might be invalid or represent no event
        if label_end <= label_start and not (
                label_start == 0.0 and label_end == 0.0):  # Invalid label interval (not the [0,0] case)
            iou_scores.append(0.0)
            continue

        # Case: Label is [0,0] (no event), but prediction exists
        if label_start == 0.0 and label_end == 0.0 and pred_end > pred_start:
            iou_scores.append(0.0)
            continue

        # Case: Label is [0,0] (no event), and prediction is also invalid/empty
        if label_start == 0.0 and label_end == 0.0 and pred_end <= pred_start:  # or simply if pred_end <= pred_start handled above
            iou_scores.append(1.0)  # Both agree there's no valid event span
            continue

        intersection_start = max(pred_start, label_start)
        intersection_end = min(pred_end, label_end)
        intersection = max(0.0, intersection_end - intersection_start)

        union = (pred_end - pred_start) + (label_end - label_start) - intersection
        iou = intersection / union if union > 0 else 0.0
        iou_scores.append(iou)

    return np.mean(iou_scores) if iou_scores else 0.0


if __name__ == "__main__":
    feature_dir = os.path.join("..", "Dataset", "geospatial_dataset", "files_wav")
    # labels_file = os.path.join("..", "raw_data", "labels.txt") # No longer needed
    poi_json_file = os.path.join("..", "outputs", "1_poi_features.json")  # Primary source of data
    statistical_priors_dir = os.path.join("..", "outputs", "statistical_priors")
    clap_weights_path = "../CLAP_weights_2022.pth"

    print(f"Device: {device}")
    print(f"Number of POI Categories: {NUM_POI_CATEGORIES}")
    print(f"Number of Audio Labels: {NUM_AUDIO_LABELS}")

    full_dataset = AudioDataset(
        feature_dir=feature_dir,
        poi_json_file_path=poi_json_file  # Pass the poi_json_file here
    )

    if len(full_dataset) == 0:
        print("Dataset is empty. Exiting.")
        exit()

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if train_size == 0 or val_size == 0:
        print(f"Train size ({train_size}) or Val size ({val_size}) is zero. Adjust dataset split or data.")
        exit()

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=2)

    epochs = 50
    criterion = torch.nn.BCEWithLogitsLoss()

    models_to_run = {
        "Audio Only": AudioClassifier(
            num_labels=NUM_AUDIO_LABELS,
            model_weights_path=clap_weights_path,
            device=device,
            use_poi=False
        ),
        "Audio+POI (Statistical Prior)": AudioClassifier(
            num_labels=NUM_AUDIO_LABELS,
            model_weights_path=clap_weights_path,
            device=device,
            use_poi=True,
            prior_strength=0.5,  # Example strength
            priors_dir=statistical_priors_dir
        )
    }

    results_summary = {}

    try:
        print("Verifying DataLoader...")
        sample_batch = next(iter(train_loader))
        s_audio_ids, s_file_paths, s_event_spans, s_labels, s_poi_multi_hots = sample_batch
        print(f"  Sample batch loaded. Labels shape: {s_labels.shape}, POI multi-hots shape: {s_poi_multi_hots.shape}")
        if s_poi_multi_hots.shape[1] != NUM_POI_CATEGORIES:
            print(
                f"  ERROR: POI multi-hots shape is {s_poi_multi_hots.shape}, expected second dim {NUM_POI_CATEGORIES}")
    except Exception as e:
        print(f"  Error verifying DataLoader: {e}")
        print("  Ensure your dataset and collate_fn are working correctly.")
        exit()

    for model_name, model_instance in models_to_run.items():
        print(f"\n--- Training {model_name} ---")

        model_instance.to(device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_instance.parameters()), lr=1e-4,
                                     weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_mAP_for_model = 0.0

        for epoch in range(epochs):
            train_loss = train(model_instance, train_loader, criterion, optimizer)
            mAP, macro_f1, accuracy, IoU = evaluate(model_instance, val_loader)
            lr_scheduler.step()

            print(f"Epoch {epoch + 1}/{epochs} (LR: {optimizer.param_groups[0]['lr']:.6e})")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val mAP: {mAP:.4f}, F1: {macro_f1:.4f}, Acc: {accuracy:.4f}, IoU: {IoU:.4f}")

            if mAP > best_mAP_for_model:
                best_mAP_for_model = mAP
                results_summary[model_name] = {
                    'Best Epoch': epoch + 1,
                    'mAP': mAP,
                    'Macro-F1': macro_f1,
                    'Accuracy': accuracy,
                    'IoU': IoU,
                    'Train Loss (at best mAP epoch)': train_loss
                }

        print(f"--- Best results for {model_name}: {results_summary.get(model_name, 'N/A')} ---")

    if results_summary:
        df_results = pd.DataFrame.from_dict(results_summary, orient='index')
        print("\n--- Final Comparison Results ---")
        print(df_results)
        df_results.to_csv("bayesian_statistical_poi_comparison_results.csv")
    else:
        print("No results to save.")
