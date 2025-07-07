"""
Geospatial Audio Analysis using Qwen2-Audio-7B-Instruct Model

This script processes audio files with geospatial context using the Qwen2-Audio-7B-Instruct model.
It supports both single and batch processing of audio files, adding contextual information from 
Points of Interest (POI) data to enhance the audio analysis.

Usage:
    # Process a single audio file
    python main.py --single --audio_path "path/to/audio.wav"
    
    # Process all audio files in a directory
    python main.py --multiple --audio_path "path/to/directory"
    
    # Specify a different POI data file
    python main.py --poi_json "path/to/poi_data.json"
    
    # Save batch processing results to a custom location
    python main.py --multiple --output "path/to/results.json"
"""

from io import BytesIO
from urllib.request import urlopen
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
import os
import json
import logging
import librosa

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./log/audio_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def match_audio_with_prompt(audio_path, data):
    # Extract the audio name from the path
    audio_name = os.path.basename(audio_path).split(".")[0]
    matching_object = None

    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict) and obj.get("id") == audio_name:
                matching_object = obj
                break

    if not matching_object or not isinstance(matching_object, dict):
        logger.warning(f"Could not find matching information for audio: {audio_name}")
        return f"Please analyze what sounds can be heard in this audio recording."

    bbox = matching_object.get("bbox")
    poi_texts = matching_object.get("poi_texts")

    if not bbox or not poi_texts:
        logger.warning(f"Found matching object but missing bbox or POI data for audio: {audio_name}")
        return f"Please analyze what sounds can be heard in this audio recording."

    # Calculate the center of the bounding box
    lat1, lon1, lat2, lon2 = bbox
    center_lat = (float(lat1) + float(lat2)) / 2
    center_lon = (float(lon1) + float(lon2)) / 2

    # Format the POI texts for better readability
    poi_formatted = "\n".join([f"- {poi}" for poi in poi_texts])

    # Construct the prompt
    prompt = f"""This audio was recorded at GPS coordinates: {center_lat:.6f}, {center_lon:.6f} (with bounding box from {lat1}, {lon1} to {lat2}, {lon2}).
Nearby POI features include:
{poi_formatted}

Please analyze what sounds can be heard in this audio recording."""
    logger.info(f"Generated prompt for audio {audio_name}: {prompt}")
    return prompt


def process_audio_with_prompt(audio_path, prompt, processor, model):
    """Process a single audio file with the given prompt and model.
    
    Args:
        audio_path: Path to the audio file
        prompt: Text prompt to use
        processor: The processor for the model
        model: The Qwen2Audio model
        
    Returns:
        Generated text response or error message
    """
    
    try:
        # Create conversation in the chat template format
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant specialized in audio analysis. Provide detailed descriptions of sounds in audio recordings, considering any geographical context provided.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ]},
        ]
        
        # Process the conversation through the chat template
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # Extract audio data from conversation
        audio_data = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audio_data.append(librosa.load(ele["audio_url"], sr=processor.feature_extractor.sampling_rate)[0])

        # Prepare inputs for the model - use 'audio' instead of 'audios'
        inputs = processor(text=text, audio=audio_data, return_tensors="pt", padding=True)
        
        # Move ALL tensors to the model's device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(model.device)
        
        # Generate response
        generate_ids = model.generate(**inputs, max_new_tokens=256)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        
        # Decode the response
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing audio {audio_path}: {str(e)}")
        return f"Error processing audio: {str(e)}"



if __name__ == "__main__":
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Process audio files with Qwen2Audio model")
    parser.add_argument("--audio_path", type=str, default="Dataset/geospatial_dataset/files_wav/372596.wav", 
                        help="Path to audio file or directory")
    parser.add_argument("--poi_json", type=str, default="outputs/poi_features_2.json", 
                        help="Path to POI features JSON file")
    parser.add_argument("--output", type=str, default="outputs/audio_analysis_results.json", 
                        help="Path to save results (for multiple processing)")
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    torch.manual_seed(1234)
    
    # Set custom cache directory for downloaded models
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    # Download processor with custom cache
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=models_dir
    )

    try:
        logger.info("Attempting to load model with automatic device mapping")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",  
            load_in_8bit=True, 
            trust_remote_code=True,
            cache_dir=models_dir
        ).eval()
        logger.info(f"Model loaded with device map: {model.hf_device_map}")
    except Exception as e:
        logger.error(f"Error loading model with automatic device mapping: {str(e)}")

    # Load the JSON data for POI features
    try:
        logger.info(f"Loading POI data from {args.poi_json}")
        with open(args.poi_json, "r") as f:
            poi_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading POI data: {e}")
        poi_data = []

   
    # Process a single audio file
    logger.info(f"Processing single audio file: {args.audio_path}")
    context_prompt = match_audio_with_prompt(args.audio_path, poi_data)
    response = process_audio_with_prompt(args.audio_path, context_prompt, processor, model)
    logger.info(f"Generated response: {response}")
    