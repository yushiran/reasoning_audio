# Audio Reasoning and Processing

This project processes audio files with geospatial context using the Qwen2-Audio-7B-Instruct model. It analyzes sounds in audio recordings and provides detailed descriptions, taking into account geographical context from Points of Interest (POI) data.

## Features

- Audio analysis with state-of-the-art Qwen2-Audio model
- Integration with geospatial Points of Interest (POI) data
- Support for both single audio file and batch processing
- Detailed logging and error handling
- Command-line interface with customizable options

## System Requirements

- Python 3.9 or higher
- CUDA-compatible GPU with at least 16GB memory (recommended)
- CPU-only mode available, but significantly slower

## Installation

You can set up the environment using either UV (Universal Versioning) or Conda. Choose the method that works best for you.

### Option 1: Installation with UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. It's faster than pip and has better dependency resolution.

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
uv venv
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

### Option 2: Installation with Conda

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download/)

2. Clone this repository:

```bash
git clone https://github.com/yushiran/reasoning-audio.git
cd reasoning_audio
```

3. Create a new Conda environment:

```bash
conda create -n audio-reasoning python=3.9
conda activate audio-reasoning
```

4. Install PyTorch with CUDA support:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

5. Install remaining dependencies using pip (inside the conda environment):

```bash
pip install -r requirements.txt
```

6. If you encounter CUDA out-of-memory errors, install bitsandbytes for model quantization:

```bash
pip install bitsandbytes>=0.39.0
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

## Troubleshooting

### CUDA Out of Memory Errors

If you encounter CUDA out of memory errors, try:

1. Enable 8-bit quantization (already implemented in the code)
2. Use a smaller model variant if available
3. Process audio files one by one rather than in batch
4. Reduce the maximum generation token length

### Audio Loading Issues

If you have issues loading audio files:

1. Ensure the audio file paths are correct
2. Verify the audio files are in a supported format (WAV recommended)
3. Check the logs for specific error messages

## Development

### Developer Setup

To set up a development environment:

```bash
# Using UV
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using Conda
conda create -n audio-reasoning-dev python=3.9
conda activate audio-reasoning-dev
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
isort .
```

## Contact

For questions or issues, please open an issue on the [project GitHub repository](https://github.com/yushiran/reasoning-audio/issues).

## License

This project is licensed under the terms of the LICENSE file included in this repository.

## Acknowledgements

- [Qwen2-Audio Model](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) by Alibaba Cloud
- [Transformers library](https://github.com/huggingface/transformers) by Hugging Face
