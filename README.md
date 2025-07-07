# Audio Reasoning and Processing

This project processes audio files with geospatial context using the Qwen2-Audio-7B-Instruct model. It analyzes sounds in audio recordings and provides detailed descriptions, taking into account geographical context from Points of Interest (POI) data.

## Demo Example

Here's an example of the model's analysis compared with the original ground truth segments:

### Audio File: `372596.wav`
<video width="320" height="240" controls>
    <source src="docs/372596.mp4" type="video/mp4">
</video>


### Geospatial Context
- **GPS Coordinates**: -40.709861, 172.674541
- **Nearby POI Features**: Bench, Park, Viewpoint, Gallery, Coastline, River, Forest, etc.

### Model Analysis Result
```json
{
  "audio_path": "Dataset/geospatial_dataset/files_wav/372596.wav",
  "response": "The sound in the background is that of a bird calling and chirping throughout the duration of the audio.",
}
```

### Ground Truth Segments (Original)
```json
  "segments": [
    {
      "start": 0.1123456789,
      "end": 4.8476543211,
      "label": "Bird sounds"
    },
    {
      "start": 5.131478842,
      "end": 6.8467629421,
      "label": "Bird sounds"
    },
    {
      "start": 10.9013475692,
      "end": 13.752947519,
      "label": "Bird sounds"
    },
    {
      "start": 17.4765920572,
      "end": 21.7563734914,
      "label": "Bird sounds"
    },
    {
      "start": 27.5302645302,
      "end": 30.9840373952,
      "label": "Bird sounds"
    },
    {
      "start": 32.2423251913,
      "end": 37.2711220394,
      "label": "Bird sounds"
    },
    {
      "start": 38.9586495624,
      "end": 40.7767693149,
      "label": "Bird sounds"
    }
  ]
```

## Installation

1. First, install UV if you don't have it:

```bash
pip install uv
```

2. Clone this repository:

```bash
git clone https://github.com/yushiran/reasoning-audio.git
cd reasoning_audio
```

3. Create and activate a virtual environment:

```bash
uv sync
source .venv/bin/activate  # On Linux/macOS
# OR
.venv\Scripts\activate  # On Windows
```

4. Install dependencies:

```bash
uv pip install -r requirements.txt
```

5. Alternatively, install directly from the pyproject.toml:

```bash
uv pip install -e .
```

## Usage

### Basic Usage

Process a single audio file:

```bash
python main.py --audio_path "Dataset/geospatial_dataset/files_wav/372596.wav"
```

### Additional Options

- Specify a custom POI data file:

```bash
python main.py --poi_json "outputs/custom_poi_features.json"
```

- Save results to a custom location:

```bash
python main.py --output "outputs/my_analysis.json"
```


## License

This project is licensed under the terms of the LICENSE file included in this repository.

## Acknowledgements

- [Qwen2-Audio Model](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) by Alibaba Cloud
- [Transformers library](https://github.com/huggingface/transformers) by Hugging Face
