from io import BytesIO
from urllib.request import urlopen
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from accelerate.utils import send_to_device
import torch
import os
import json
import logging
import librosa
import glob
from app.prompts.prompt_manager import prompt_manager
from app.config import global_config

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
    # segments = matching_object.get("segments", [])

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
    prompt = prompt_manager.load(
        relative_path="07_08_Classification_Prompt.jinja2",
        center_lat=center_lat,
        center_lon=center_lon,
        lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2,
        poi_formatted=poi_formatted
    )
    # logger.info(f"Generated prompt for audio {audio_name}: {prompt}")
    return prompt, matching_object


def process_audio_with_prompt(audio_path, prompt, processor, model, output_dir=None, matching_object=None):
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
            {'role': 'system', 'content': 'You are a helpful assistant specialized in audio classification.'},
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

        inputs = processor(text=text, audio=audio_data, return_tensors="pt", padding=True)
      
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(model.device)


        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=256)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        
        # Decode the response
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Save output to file if output_path is provided
        if output_dir:
            try:
                # Create directory if it doesn't exist
                output_path = f"{output_dir}/{matching_object.get('id', 'unknown')}.json"
                os.makedirs(os.path.dirname(f"{output_path}"), exist_ok=True)

                # Prepare result data
                result_data = {
                    "audio_path": audio_path,
                    "prompt": prompt,
                    "response": response,
                    "segments": matching_object.get("segments", []) if matching_object else [],
                }
                
                # Save to JSON file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)

                logger.info(f"Results saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving results to {output_path}: {str(e)}")

        return response
        
    except Exception as e:
        logger.error(f"Error processing audio {audio_path}: {str(e)}")
        return f"Error processing audio: {str(e)}"



if __name__ == "__main__":
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Process audio files with Qwen2Audio model")
    parser.add_argument("--audio_path", type=str, default="Dataset/500_Geo/Geo_500", 
                        help="Path to audio file or directory")
    parser.add_argument("--poi_json", type=str, default="Dataset/500_Geo/Geo_500.json", 
                        help="Path to POI features JSON file")
    parser.add_argument("--output", type=str, default="outputs/07_08_classification", 
                        help="Path to save results (for multiple processing)")
    args = parser.parse_args()
        
    torch.manual_seed(1234)
    
    # Set custom cache directory for downloaded models
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    # Download processor with custom cache
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=global_config.MODEL_PATH
    )

    try:
        logger.info("Attempting to load model with automatic device mapping")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            device_map="cuda:1",  
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance on GPUs
            trust_remote_code=True,
            cache_dir=global_config.MODEL_PATH
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
    logger.info(f"Processing audio files in directory: {args.audio_path}")
    audio_files = glob.glob(os.path.join(args.audio_path, "*.wav"))
    for audio_file in audio_files:
        context_prompt, matching_object = match_audio_with_prompt(audio_file, poi_data)
        response = process_audio_with_prompt(audio_file, context_prompt, processor, model, args.output, matching_object)
        logger.info(f"Generated response for {audio_file}")
