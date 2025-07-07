# Copilot Instructions for Audio Reasoning Project

## Project Overview
This project implements a Chain-of-Thought (CoT) reasoning framework for audio analysis with geospatial context. The system processes audio files along with geographical information to perform structured reasoning about potential events and scenarios.

## Current Framework Architecture

### Input Components
1. **Audio Data**: Raw audio files (WAV format) containing various sound events
2. **Geospatial Text Description**: Contextual information including:
   - GPS coordinates
   - Points of Interest (POI) features
   - Environmental context (parks, coastlines, urban areas, etc.)
3. **Reasoning Prompt**: Structured prompts like:
   ```
   "Please combine the original audio information with the geographical text description, 
   think step by step, and analyze the possible events that may occur."
   ```

### Model Architecture
- **Base Model**: Qwen2-Audio-7B-Instruct
- **Future Enhancement**: GRPO (Group Relative Policy Optimization) fine-tuning
- **Reasoning Method**: Chain-of-Thought (CoT) integration

### Output Format
- **Reasoning Process**: Step-by-step analytical thinking
- **Final Answer**: Structured conclusion about identified events

## Development Guidelines

### Code Structure Priorities
1. **Modular Design**: Separate components for audio processing, geospatial context, and reasoning
2. **Pipeline Architecture**: Clear input → processing → reasoning → output flow
3. **Extensibility**: Prepare for future GRPO fine-tuning integration

### Key Implementation Areas
1. **Audio Processing Module**
   - Audio file loading and preprocessing
   - Feature extraction preparation
   - Integration with Qwen2-Audio model

2. **Geospatial Context Handler**
   - POI data processing
   - Geographic context formatting
   - Text description generation

3. **Reasoning Engine**
   - Prompt engineering for CoT
   - Step-by-step reasoning structure
   - Output formatting and validation

4. **Evaluation Framework**
   - Prepare for benchmark testing (MMAU, AIR-Bench)
   - Reasoning quality assessment
   - Performance metrics tracking

### Technical Considerations
- **Memory Management**: Efficient handling of audio data and model inference
- **Prompt Engineering**: Optimize prompts for geospatial-audio reasoning
- **Error Handling**: Robust processing for various audio formats and quality
- **Logging**: Comprehensive logging for debugging and analysis

### Future Development Path
1. **Phase 1**: Basic framework implementation (current focus)
2. **Phase 2**: GRPO fine-tuning integration
3. **Phase 3**: Advanced CoT reasoning enhancement
4. **Phase 4**: Evaluation and benchmarking

## File Organization Preferences
- `main.py`: Primary processing pipeline
- `audio_processor/`: Audio handling modules
- `geospatial/`: Geographic context processing
- `reasoning/`: CoT reasoning implementation
- `models/`: Model management and inference
- `evaluation/`: Testing and benchmarking tools

## Reference Research
This project builds upon cutting-edge research in audio reasoning:
- **GAMA (2024)**: CompA-R dataset and Audio Q-Former architecture
- **Audio-CoT (2025)**: CoT prompting for audio tasks
- **SARI (2025)**: Structured reasoning with reinforcement learning
- **Audio-Reasoner (2025)**: Large-scale audio reasoning datasets

See `reference/reference.md` for detailed analysis of related work.

## Coding Assistance Focus
When providing code suggestions or implementations:
1. Prioritize the three-component input structure (audio + geo + prompt)
2. Design with future GRPO fine-tuning in mind
3. Implement clear reasoning chain outputs
4. Ensure compatibility with Qwen2-Audio model requirements
5. Maintain extensibility for advanced CoT methods