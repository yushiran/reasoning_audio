import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
import json
import os
import pandas as pd
import torch.nn as nn
from CLAP.CLAPWrapper import CLAPWrapper
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
POI_BERT_EMBED_DIM = 768
CLAP_EMBED_DIM = 1024


class AudioDataset_Bayesian(Dataset):
    def __init__(self, feature_dir, poi_json_file_path):
        self.feature_dir = feature_dir
        self.all_data = []
        self.audio_ids = []
        self.audio_id_to_poi_hot = {}

        try:
            with open(poi_json_file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            for item in loaded_data:
                if 'id' not in item:
                    continue
                audio_id_str = str(item['id'])
                if 'audio_spans' not in item:
                    item['audio_spans'] = []
                self.all_data.append(item)
                self.audio_ids.append(audio_id_str)
                if 'poi_multi_hot' in item and isinstance(item['poi_multi_hot'], list) and \
                        len(item['poi_multi_hot']) == NUM_POI_CATEGORIES:
                    self.audio_id_to_poi_hot[audio_id_str] = item['poi_multi_hot']
        except FileNotFoundError:
            print(f"Error: POI JSON file not found at {poi_json_file_path}.")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error reading POI JSON from {poi_json_file_path}: {e}.")
        self.label_map = AUDIO_LABEL_TO_IDX

    def __len__(self):
        return len(self.audio_ids)

    def __getitem__(self, idx):
        audio_id = self.audio_ids[idx]
        current_item_data = self.all_data[idx]
        wav_path = os.path.join(self.feature_dir, f"{audio_id}.wav")
        labels = torch.zeros(NUM_AUDIO_LABELS, dtype=torch.float)
        event_spans_for_item = []
        for span in current_item_data.get("audio_spans", []):
            label = span["label"]
            if label in self.label_map:
                labels[self.label_map[label]] = 1.0
            event_spans_for_item.append({"start": span["start"], "end": span["end"], "label": span["label"]})
        poi_multi_hot_list = self.audio_id_to_poi_hot.get(audio_id, [0] * NUM_POI_CATEGORIES)
        poi_multi_hot = torch.tensor(poi_multi_hot_list, dtype=torch.float)
        return audio_id, wav_path, event_spans_for_item, labels, poi_multi_hot


def collate_fn_bayesian(batch):
    audio_ids, file_paths, event_spans, labels, poi_multi_hots = zip(*batch)
    return list(audio_ids), list(file_paths), list(event_spans), torch.stack(labels, 0), torch.stack(poi_multi_hots, 0)


class AudioDataset_EmbeddingPOI(Dataset):
    def __init__(self, feature_dir, poi_json_file_path, poi_feat_dir):
        self.feature_dir = feature_dir
        self.poi_feat_dir = poi_feat_dir
        self.all_data = []
        self.audio_ids = []
        try:
            with open(poi_json_file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            for item in loaded_data:
                if 'id' not in item:
                    continue
                audio_id_str = str(item['id'])
                if 'audio_spans' not in item:
                    item['audio_spans'] = []
                self.all_data.append(item)
                self.audio_ids.append(audio_id_str)
        except FileNotFoundError:
            print(f"Error: POI JSON file not found at {poi_json_file_path}.")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error reading POI JSON from {poi_json_file_path}: {e}.")
        self.label_map = AUDIO_LABEL_TO_IDX

    def __len__(self):
        return len(self.audio_ids)

    def __getitem__(self, idx):
        audio_id = self.audio_ids[idx]
        current_item_data = self.all_data[idx]
        wav_path = os.path.join(self.feature_dir, f"{audio_id}.wav")
        labels = torch.zeros(NUM_AUDIO_LABELS, dtype=torch.float)
        event_spans_for_item = []
        for span in current_item_data.get("audio_spans", []):
            label = span["label"]
            if label in self.label_map:
                labels[self.label_map[label]] = 1.0
            event_spans_for_item.append({"start": span["start"], "end": span["end"], "label": span["label"]})
        poi_embedding_path = os.path.join(self.poi_feat_dir, f"{audio_id}_poi_cls_embeddings.npy")
        poi_embedding = torch.zeros(POI_BERT_EMBED_DIM, dtype=torch.float)
        try:
            if os.path.exists(poi_embedding_path):
                embedding_data = np.load(poi_embedding_path)
                if embedding_data.shape == (POI_BERT_EMBED_DIM,):
                    poi_embedding = torch.tensor(embedding_data, dtype=torch.float)
        except Exception as e:
            print(f"Error loading POI embedding for {audio_id} from {poi_embedding_path}: {e}. Using zeros.")
        return audio_id, wav_path, event_spans_for_item, labels, poi_embedding


def collate_fn_embedding_poi(batch):
    audio_ids, file_paths, event_spans, labels, poi_embeddings = zip(*batch)
    return list(audio_ids), list(file_paths), list(event_spans), torch.stack(labels, 0), torch.stack(poi_embeddings, 0)


class AudioClassifier_Bayesian(nn.Module):
    def __init__(self, num_labels, model_weights_path, device, use_poi=False, prior_strength=0.5, priors_dir=None):
        super().__init__()
        self.use_poi = use_poi
        self.device = device
        self.prior_strength = prior_strength
        self.num_labels = num_labels
        self.clap_wrapper = CLAPWrapper(model_fp=model_weights_path, device=device)
        self.audio_event_classifier = nn.Sequential(
            nn.Linear(CLAP_EMBED_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        ).to(device)
        if self.use_poi:
            if priors_dir is None: raise ValueError("priors_dir must be provided for Bayesian model with use_poi=True")
            try:
                self.statistical_poi_event_priors = torch.load(
                    os.path.join(priors_dir, "statistical_poi_event_priors.pt")).to(device)
                self.statistical_marginal_event_priors = torch.load(
                    os.path.join(priors_dir, "statistical_marginal_event_priors.pt")).to(device)
                if self.statistical_poi_event_priors.shape != (NUM_POI_CATEGORIES, num_labels) or \
                        self.statistical_marginal_event_priors.shape != (num_labels,):
                    raise ValueError("Loaded statistical priors have incorrect shapes.")
            except Exception as e:
                print(
                    f"Error loading priors for Bayesian model from {priors_dir}: {e}. POI integration will be disabled.")
                self.use_poi = False
        self.time_predictor = nn.Linear(CLAP_EMBED_DIM, 2).to(device)

    def forward(self, file_paths: list, batch_poi_multi_hot=None):
        audio_embeddings = self.clap_wrapper.get_audio_embeddings(file_paths, resample=True)
        if audio_embeddings.dim() == 3: audio_embeddings = audio_embeddings.mean(dim=1)
        audio_logits = self.audio_event_classifier(audio_embeddings.to(torch.float32))
        time_outputs = self.time_predictor(audio_embeddings.to(torch.float32))
        final_logits = audio_logits
        if self.use_poi and batch_poi_multi_hot is not None and \
                hasattr(self, 'statistical_poi_event_priors') and hasattr(self, 'statistical_marginal_event_priors'):
            batch_poi_multi_hot = batch_poi_multi_hot.to(self.device)
            prior_logits_list = []
            for i in range(batch_poi_multi_hot.size(0)):
                sample_poi_hot = batch_poi_multi_hot[i]
                active_poi_indices = sample_poi_hot.nonzero(as_tuple=True)[0]
                sample_prior_probs = self.statistical_marginal_event_priors if len(active_poi_indices) == 0 \
                    else self.statistical_poi_event_priors[active_poi_indices, :].mean(dim=0)
                prior_logits_list.append(torch.log(sample_prior_probs + 1e-9))
            if prior_logits_list:
                prior_logits = torch.stack(prior_logits_list)
                final_logits = audio_logits + self.prior_strength * prior_logits
        return final_logits, time_outputs


class POIPriorNet(nn.Module):
    def __init__(self, poi_embed_dim, projected_audio_dim, hidden_dim=256):
        super().__init__()
        self.prior_net = nn.Sequential(
            nn.Linear(poi_embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, projected_audio_dim)
        )

    def forward(self, poi_bert_embeddings):
        return self.prior_net(poi_bert_embeddings)


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate,
                                                    batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(embed_dim * 2, embed_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, audio_query_features, poi_context_features):
        audio_q = audio_query_features.unsqueeze(1)
        poi_kv = poi_context_features.unsqueeze(1)
        attn_output, _ = self.multihead_attn(query=audio_q, key=poi_kv, value=poi_kv)
        fused = self.norm1(audio_query_features + attn_output.squeeze(1))
        fused = self.dropout(fused)
        ffn_output = self.ffn(fused)
        fused = self.norm2(fused + ffn_output)
        fused = self.dropout(fused)
        return fused


class AudioClassifier_CrossAttention(nn.Module):
    def __init__(self, num_labels, model_weights_path, device, use_poi=False,
                 poi_embed_dim=POI_BERT_EMBED_DIM, projected_audio_dim=512,
                 poi_hidden_dim=256, num_attn_heads=4, dropout_rate=0.1):
        super().__init__()
        self.use_poi = use_poi
        self.device = device
        self.num_labels = num_labels
        self.projected_audio_dim = projected_audio_dim
        self.clap_wrapper = CLAPWrapper(model_fp=model_weights_path, device=device)
        self.audio_feature_projector = nn.Sequential(
            nn.Linear(CLAP_EMBED_DIM, self.projected_audio_dim),
            nn.BatchNorm1d(self.projected_audio_dim), nn.ReLU()
        ).to(device)
        self.audio_only_classifier_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.projected_audio_dim, num_labels)
        ).to(device)
        if self.use_poi:
            self.poi_processor = POIPriorNet(
                poi_embed_dim=poi_embed_dim,
                projected_audio_dim=self.projected_audio_dim,
                hidden_dim=poi_hidden_dim
            ).to(device)
            self.fusion_module = CrossAttentionFusion(
                embed_dim=self.projected_audio_dim,
                num_heads=num_attn_heads, dropout_rate=dropout_rate
            ).to(device)
            self.fusion_classifier_head = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(self.projected_audio_dim, num_labels)
            ).to(device)
        self.time_predictor = nn.Linear(CLAP_EMBED_DIM, 2).to(device)

    def forward(self, file_paths: list, batch_poi_embeddings=None):
        audio_clap_embeddings = self.clap_wrapper.get_audio_embeddings(file_paths, resample=True)
        if audio_clap_embeddings.dim() == 3:
            audio_clap_embeddings = audio_clap_embeddings.mean(dim=1)
        audio_clap_embeddings_float = audio_clap_embeddings.to(torch.float32)
        projected_audio_features = self.audio_feature_projector(audio_clap_embeddings_float)
        time_outputs = self.time_predictor(audio_clap_embeddings_float)
        if self.use_poi and batch_poi_embeddings is not None:
            if torch.all(batch_poi_embeddings == 0) or \
                    (batch_poi_embeddings.shape[0] > 0 and batch_poi_embeddings.shape[1] != POI_BERT_EMBED_DIM):
                final_logits = self.audio_only_classifier_head(projected_audio_features)
            else:
                processed_poi_features = self.poi_processor(batch_poi_embeddings.to(self.device))
                fused_features = self.fusion_module(projected_audio_features, processed_poi_features)
                final_logits = self.fusion_classifier_head(fused_features)
        else:
            final_logits = self.audio_only_classifier_head(projected_audio_features)
        return final_logits, time_outputs


class POIFiLMParameterGenerator(nn.Module):
    def __init__(self, poi_embed_dim, audio_feature_dim, hidden_dim=256):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(poi_embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * audio_feature_dim)
        )
        self.audio_feature_dim = audio_feature_dim

    def forward(self, poi_embeddings):
        params = self.generator(poi_embeddings)
        gamma = params[:, :self.audio_feature_dim]
        beta = params[:, self.audio_feature_dim:]
        return gamma, beta


class AudioClassifier_FiLM(nn.Module):
    def __init__(self, num_labels, model_weights_path, device, use_poi=False,
                 poi_embed_dim=POI_BERT_EMBED_DIM, projected_audio_dim=512,
                 film_poi_hidden_dim=256):
        super().__init__()
        self.use_poi = use_poi
        self.device = device
        self.num_labels = num_labels
        self.projected_audio_dim = projected_audio_dim
        self.clap_wrapper = CLAPWrapper(model_fp=model_weights_path, device=device)
        self.audio_feature_projector = nn.Sequential(
            nn.Linear(CLAP_EMBED_DIM, self.projected_audio_dim),
            nn.BatchNorm1d(self.projected_audio_dim), nn.ReLU()
        ).to(device)
        self.audio_only_classifier_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.projected_audio_dim, num_labels)
        ).to(device)
        if self.use_poi:
            self.poi_film_param_generator = POIFiLMParameterGenerator(
                poi_embed_dim=poi_embed_dim,
                audio_feature_dim=self.projected_audio_dim,
                hidden_dim=film_poi_hidden_dim
            ).to(device)
            self.fusion_classifier_head = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(self.projected_audio_dim, num_labels)
            ).to(device)
        self.time_predictor = nn.Linear(CLAP_EMBED_DIM, 2).to(device)

    def forward(self, file_paths: list, batch_poi_embeddings=None):
        audio_clap_embeddings = self.clap_wrapper.get_audio_embeddings(file_paths, resample=True)
        if audio_clap_embeddings.dim() == 3:
            audio_clap_embeddings = audio_clap_embeddings.mean(dim=1)
        audio_clap_embeddings_float = audio_clap_embeddings.to(torch.float32)
        projected_audio_features = self.audio_feature_projector(audio_clap_embeddings_float)
        time_outputs = self.time_predictor(audio_clap_embeddings_float)
        if self.use_poi and batch_poi_embeddings is not None:
            if torch.all(batch_poi_embeddings == 0) or \
                    (batch_poi_embeddings.shape[0] > 0 and batch_poi_embeddings.shape[1] != POI_BERT_EMBED_DIM):
                final_logits = self.audio_only_classifier_head(projected_audio_features)
            else:
                gamma, beta = self.poi_film_param_generator(batch_poi_embeddings.to(self.device))
                modulated_features = gamma * projected_audio_features + beta
                final_logits = self.fusion_classifier_head(modulated_features)
        else:
            final_logits = self.audio_only_classifier_head(projected_audio_features)
        return final_logits, time_outputs


class POITransformerProcessor(nn.Module):
    def __init__(self, poi_embed_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.processor = nn.Sequential(
            nn.Linear(poi_embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, poi_embeddings):
        return self.processor(poi_embeddings)


class TransformerFusionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=2, dropout_rate=0.1, dim_feedforward=None):
        super().__init__()
        if dim_feedforward is None: dim_feedforward = embed_dim * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout_rate, batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_dim = embed_dim

    def forward(self, audio_features, poi_features):
        sequence = torch.stack([audio_features, poi_features], dim=1)
        transformer_output = self.transformer_encoder(sequence)
        return transformer_output[:, 0, :]


class AudioClassifier_TransformerFusion(nn.Module):
    def __init__(self, num_labels, model_weights_path, device, use_poi=False,
                 poi_embed_dim=POI_BERT_EMBED_DIM, projected_audio_dim=512,
                 poi_transformer_hidden_dim=256,
                 transformer_num_heads=4, transformer_num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.use_poi = use_poi
        self.device = device
        self.num_labels = num_labels
        self.projected_audio_dim = projected_audio_dim
        self.clap_wrapper = CLAPWrapper(model_fp=model_weights_path, device=device)
        self.audio_feature_projector = nn.Sequential(
            nn.Linear(CLAP_EMBED_DIM, self.projected_audio_dim),
            nn.BatchNorm1d(self.projected_audio_dim), nn.ReLU()
        ).to(device)
        self.audio_only_classifier_head = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(self.projected_audio_dim, num_labels)
        ).to(device)
        if self.use_poi:
            self.poi_transformer_processor = POITransformerProcessor(
                poi_embed_dim=poi_embed_dim, output_dim=self.projected_audio_dim,
                hidden_dim=poi_transformer_hidden_dim
            ).to(device)
            self.transformer_fusion_layer = TransformerFusionLayer(
                embed_dim=self.projected_audio_dim, num_heads=transformer_num_heads,
                num_layers=transformer_num_layers, dropout_rate=dropout_rate
            ).to(device)
            self.fusion_classifier_head = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(self.projected_audio_dim, num_labels)
            ).to(device)
        self.time_predictor = nn.Linear(CLAP_EMBED_DIM, 2).to(device)

    def forward(self, file_paths: list, batch_poi_embeddings=None):
        audio_clap_embeddings = self.clap_wrapper.get_audio_embeddings(file_paths, resample=True)
        if audio_clap_embeddings.dim() == 3:
            audio_clap_embeddings = audio_clap_embeddings.mean(dim=1)
        audio_clap_embeddings_float = audio_clap_embeddings.to(torch.float32)
        projected_audio_features = self.audio_feature_projector(audio_clap_embeddings_float)
        time_outputs = self.time_predictor(audio_clap_embeddings_float)
        if self.use_poi and batch_poi_embeddings is not None:
            if torch.all(batch_poi_embeddings == 0) or \
                    (batch_poi_embeddings.shape[0] > 0 and batch_poi_embeddings.shape[1] != POI_BERT_EMBED_DIM):
                final_logits = self.audio_only_classifier_head(projected_audio_features)
            else:
                processed_poi_features = self.poi_transformer_processor(batch_poi_embeddings.to(self.device))
                fused_features = self.transformer_fusion_layer(projected_audio_features, processed_poi_features)
                final_logits = self.fusion_classifier_head(fused_features)
        else:
            final_logits = self.audio_only_classifier_head(projected_audio_features)
        return final_logits, time_outputs


def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    time_criterion = nn.MSELoss()
    for batch_data in dataloader:
        _, file_paths, event_spans, labels, poi_data = batch_data
        labels = labels.to(model.device)
        poi_data = poi_data.to(model.device)
        time_labels_list = []
        for spans in event_spans:
            time_labels_list.append([spans[0]['start'], spans[0]['end']] if len(spans) > 0 else [0.0, 0.0])
        time_labels = torch.tensor(time_labels_list, dtype=torch.float32).to(model.device)

        if isinstance(model, AudioClassifier_Bayesian):
            outputs, time_outputs = model(file_paths, batch_poi_multi_hot=poi_data if model.use_poi else None)
        elif isinstance(model,
                        (AudioClassifier_CrossAttention, AudioClassifier_FiLM, AudioClassifier_TransformerFusion)):
            outputs, time_outputs = model(file_paths, batch_poi_embeddings=poi_data if model.use_poi else None)
        else:
            raise ValueError(f"Unknown model instance type: {type(model)} for training.")

        class_loss = criterion(outputs, labels)
        time_loss = time_criterion(time_outputs, time_labels)
        total_loss = class_loss + 0.3 * time_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
    return running_loss / len(dataloader)


def evaluate_model(model, dataloader):
    model.eval()
    all_preds_scores, all_preds_binary, all_labels = [], [], []
    all_time_preds, all_time_labels = [], []
    with torch.no_grad():
        for batch_data in dataloader:
            _, file_paths, event_spans, labels, poi_data = batch_data
            labels_device = labels.to(model.device)
            poi_data_device = poi_data.to(model.device)

            if isinstance(model, AudioClassifier_Bayesian):
                outputs, time_outputs = model(file_paths,
                                              batch_poi_multi_hot=poi_data_device if model.use_poi else None)
            elif isinstance(model,
                            (AudioClassifier_CrossAttention, AudioClassifier_FiLM, AudioClassifier_TransformerFusion)):
                outputs, time_outputs = model(file_paths,
                                              batch_poi_embeddings=poi_data_device if model.use_poi else None)
            else:
                raise ValueError(f"Unknown model instance type: {type(model)} for evaluation.")

            preds_scores = torch.sigmoid(outputs).cpu().numpy()
            preds_binary = (preds_scores > 0.5).astype(float)
            all_preds_scores.append(preds_scores)
            all_preds_binary.append(preds_binary)
            all_labels.append(labels_device.cpu().numpy())
            all_time_preds.append(time_outputs.cpu().numpy())
            time_labels_list = []
            for spans in event_spans:
                time_labels_list.append([spans[0]['start'], spans[0]['end']] if len(spans) > 0 else [0.0, 0.0])
            all_time_labels.append(np.array(time_labels_list, dtype=np.float32))

    all_preds_scores = np.vstack(all_preds_scores)
    all_preds_binary = np.vstack(all_preds_binary)
    all_labels = np.vstack(all_labels)
    accuracy, f1_macro, f1_micro, mean_ap = 0.0, 0.0, 0.0, 0.0

    if np.sum(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_preds_binary)
        f1_macro = f1_score(all_labels, all_preds_binary, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds_binary, average='micro', zero_division=0)
        ap_scores = [average_precision_score(all_labels[:, i], all_preds_scores[:, i])
                     for i in range(NUM_AUDIO_LABELS) if np.sum(all_labels[:, i]) > 0]
        if ap_scores:
            mean_ap = np.nanmean([s for s in ap_scores if not np.isnan(s)])
            if np.isnan(mean_ap): mean_ap = 0.0
    all_time_preds = np.vstack(all_time_preds)
    all_time_labels = np.vstack(all_time_labels)
    time_mse = np.mean((all_time_preds - all_time_labels) ** 2)
    return accuracy, f1_macro, f1_micro, mean_ap, time_mse


def main():
    clap_model_path = os.path.join("..", "CLAP_weights_2022.pth")
    audio_features_dir = os.path.join("..", "Dataset", "geospatial_dataset", "files_wav")
    poi_and_labels_json_file = os.path.join("..", "outputs", "1_poi_features.json")
    poi_bayesian_priors_dir = os.path.join("..", "outputs", "statistical_priors")
    poi_feat_dir = os.path.join("..", "outputs", "poi_features")
    output_results_dir = os.path.join("..", "outputs", "results_all_CLAP")
    os.makedirs(output_results_dir, exist_ok=True)

    epochs = 100
    batch_size = 16
    learning_rate = 1e-4
    weight_decay_val = 1e-5

    experiments = {
        "bayesian_audio_only": {
            "model_class": AudioClassifier_Bayesian, "dataset_class": AudioDataset_Bayesian,
            "collate_fn": collate_fn_bayesian, "use_poi": False,
            "model_params": {"prior_strength": 0.5, "priors_dir": None},
            "dataset_params": {"poi_json_file_path": poi_and_labels_json_file}
        },
        "bayesian_fusion": {
            "model_class": AudioClassifier_Bayesian, "dataset_class": AudioDataset_Bayesian,
            "collate_fn": collate_fn_bayesian, "use_poi": True,
            "model_params": {"prior_strength": 0.5, "priors_dir": poi_bayesian_priors_dir},
            "dataset_params": {"poi_json_file_path": poi_and_labels_json_file}
        },
        "cross_attention_fusion": {
            "model_class": AudioClassifier_CrossAttention, "dataset_class": AudioDataset_EmbeddingPOI,
            "collate_fn": collate_fn_embedding_poi, "use_poi": True,
            "model_params": {"projected_audio_dim": 512, "poi_embed_dim": POI_BERT_EMBED_DIM},
            "dataset_params": {"poi_json_file_path": poi_and_labels_json_file, "poi_feat_dir": poi_feat_dir}
        },
        "film_fusion": {
            "model_class": AudioClassifier_FiLM, "dataset_class": AudioDataset_EmbeddingPOI,
            "collate_fn": collate_fn_embedding_poi, "use_poi": True,
            "model_params": {"projected_audio_dim": 512, "poi_embed_dim": POI_BERT_EMBED_DIM},
            "dataset_params": {"poi_json_file_path": poi_and_labels_json_file, "poi_feat_dir": poi_feat_dir}
        },
        "transformer_fusion": {
            "model_class": AudioClassifier_TransformerFusion, "dataset_class": AudioDataset_EmbeddingPOI,
            "collate_fn": collate_fn_embedding_poi, "use_poi": True,
            "model_params": {"projected_audio_dim": 512, "poi_embed_dim": POI_BERT_EMBED_DIM},
            "dataset_params": {"poi_json_file_path": poi_and_labels_json_file, "poi_feat_dir": poi_feat_dir}
        }
    }
    results = {}
    for exp_name, config in experiments.items():
        print(f"\n--- Running Experiment: {exp_name} ---")
        full_dataset_constructor_params = {"feature_dir": audio_features_dir, **config["dataset_params"]}
        full_dataset = config["dataset_class"](**full_dataset_constructor_params)

        if len(full_dataset) == 0:
            print(f"Dataset for {exp_name} is empty. Skipping.")
            results[exp_name] = {"status": "skipped_dataset_empty", "error": "Dataset empty"}
            continue

        total_len = len(full_dataset)
        train_len = int(0.7 * total_len)
        val_len = int(0.15 * total_len)
        test_len = total_len - train_len - val_len

        if train_len == 0 or val_len == 0 or test_len == 0:
            print(f"Dataset for {exp_name} too small for split. Skipping.")
            results[exp_name] = {"status": "skipped_dataset_too_small", "error": "Dataset too small"}
            continue

        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=config["collate_fn"],
                                  num_workers=2, pin_memory=device.type == 'cuda')
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=config["collate_fn"],
                                num_workers=2, pin_memory=device.type == 'cuda')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=config["collate_fn"],
                                 num_workers=2, pin_memory=device.type == 'cuda')

        model = config["model_class"](
            num_labels=NUM_AUDIO_LABELS, model_weights_path=clap_model_path, device=device,
            use_poi=config["use_poi"], **config["model_params"]
        )
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                                     weight_decay=weight_decay_val)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best_val_mAP = -1.0
        epoch_of_best_val_mAP = 0
        best_model_path = os.path.join(output_results_dir, f"{exp_name}_best_val_model.pt")

        for epoch in range(epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer)
            val_accuracy, val_f1_macro, val_f1_micro, val_mean_ap, val_time_mse = evaluate_model(model, val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{epochs} - LR: {current_lr:.1e} - Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | Val F1 (Ma): {val_f1_macro:.4f} | Val F1 (Mi): {val_f1_micro:.4f} | "
                f"Val mAP: {val_mean_ap:.4f} | Val TimeMSE: {val_time_mse:.4f}")
            if val_mean_ap > best_val_mAP:
                best_val_mAP = val_mean_ap
                epoch_of_best_val_mAP = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                print(f"  --> Saved new best model for {exp_name} (Epoch {epoch + 1}) with Val mAP: {best_val_mAP:.4f}")
            lr_scheduler.step()

        final_model_path = os.path.join(output_results_dir, f"{exp_name}_final_epoch_model.pt")
        torch.save(model.state_dict(), final_model_path)

        print(f"\n--- Evaluating Final Epoch Model for {exp_name} on Test Set ---")
        (final_test_accuracy, final_test_f1_macro, final_test_f1_micro,
         final_test_mean_ap, final_test_time_mse) = evaluate_model(model, test_loader)
        print(
            f"Final Test: Acc: {final_test_accuracy:.4f}, F1 (Ma): {final_test_f1_macro:.4f}, "
            f"F1 (Mi): {final_test_f1_micro:.4f}, mAP: {final_test_mean_ap:.4f}, TimeMSE: {final_test_time_mse:.4f}")
        metrics_final_epoch_model_on_test = {
            "test_accuracy": float(final_test_accuracy), "test_f1_macro": float(final_test_f1_macro),
            "test_f1_micro": float(final_test_f1_micro), "test_mean_ap": float(final_test_mean_ap),
            "test_time_mse": float(final_test_time_mse)
        }
        metrics_best_val_model_on_test = {}
        if epoch_of_best_val_mAP > 0 and os.path.exists(best_model_path):
            print(
                f"\n--- Evaluating Best Validation Model (Epoch {epoch_of_best_val_mAP}) for {exp_name} on Test Set ---")
            model.load_state_dict(torch.load(best_model_path))
            (best_val_test_accuracy, best_val_test_f1_macro, best_val_test_f1_micro,
             best_val_test_mean_ap, best_val_test_time_mse) = evaluate_model(model, test_loader)
            print(
                f"Best Val Model Test: Acc: {best_val_test_accuracy:.4f}, F1 (Ma): {best_val_test_f1_macro:.4f}, "
                f"F1 (Mi): {best_val_test_f1_micro:.4f}, mAP: {best_val_test_mean_ap:.4f}, TimeMSE: {best_val_test_time_mse:.4f}")
            metrics_best_val_model_on_test = {
                "test_accuracy": float(best_val_test_accuracy), "test_f1_macro": float(best_val_test_f1_macro),
                "test_f1_micro": float(best_val_test_f1_micro), "test_mean_ap": float(best_val_test_mean_ap),
                "test_time_mse": float(best_val_test_time_mse)
            }
        else:
            metrics_best_val_model_on_test = {k: float('nan') for k in metrics_final_epoch_model_on_test}
        results[exp_name] = {
            "status": "completed",
            "best_val_mAP_score_on_val_set": float(best_val_mAP),
            "epoch_of_best_val_mAP": int(epoch_of_best_val_mAP),
            "metrics_best_val_model_on_test": metrics_best_val_model_on_test,
            "metrics_final_epoch_model_on_test": metrics_final_epoch_model_on_test
        }

    results_filepath = os.path.join(output_results_dir, "all_fusion_experiment_results.json")
    with open(results_filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nAll experiment results saved to {results_filepath}")

    flat_results_for_csv = []
    metric_keys_from_eval = ["test_accuracy", "test_f1_macro", "test_f1_micro", "test_mean_ap", "test_time_mse"]
    for exp_name_key, data_dict in results.items():
        row = {"experiment_name": exp_name_key, "status": data_dict.get("status", "unknown")}
        if data_dict.get("status") == "completed":
            row["best_val_mAP_score_on_val_set"] = data_dict.get("best_val_mAP_score_on_val_set")
            row["epoch_of_best_val_mAP"] = data_dict.get("epoch_of_best_val_mAP")
            metrics_best = data_dict.get("metrics_best_val_model_on_test", {})
            for metric_k in metric_keys_from_eval: row[f"best_val_model_{metric_k}"] = metrics_best.get(metric_k,
                                                                                                        float('nan'))
            metrics_final = data_dict.get("metrics_final_epoch_model_on_test", {})
            for metric_k in metric_keys_from_eval: row[f"final_epoch_model_{metric_k}"] = metrics_final.get(metric_k,
                                                                                                            float(
                                                                                                                'nan'))
        elif "error" in data_dict:
            row["error"] = data_dict.get("error")
        flat_results_for_csv.append(row)

    try:
        df_results = pd.DataFrame(flat_results_for_csv)
        preferred_cols_order = ["experiment_name", "status", "best_val_mAP_score_on_val_set", "epoch_of_best_val_mAP"]
        for prefix in ["best_val_model_", "final_epoch_model_"]:
            for mn in metric_keys_from_eval: preferred_cols_order.append(f"{prefix}{mn}")
        if "error" in df_results.columns: preferred_cols_order.append("error")
        existing_cols_in_order = [col for col in preferred_cols_order if col in df_results.columns]
        other_cols = [col for col in df_results.columns if col not in existing_cols_in_order]
        df_results = df_results[existing_cols_in_order + other_cols]
        csv_path = os.path.join(output_results_dir, "all_fusion_experiment_results.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"Results also saved to CSV: {csv_path}")
    except Exception as e:
        print(f"Could not save results to CSV: {e}")


if __name__ == "__main__":
    main()
