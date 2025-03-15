# Voice AI Agent

An AI agent that generates high-quality audio output from text prompts with voice cloning capabilities. Supports deployment on RunPod serverless for GPU-accelerated inference.

## Features

- 🔊 Natural and expressive speech synthesis using multiple TTS models:
  - **Kokoro-82M**: Lightweight yet expressive multilingual TTS model
  - **Zonos-v0.1-hybrid**: Advanced voice cloning with improved quality
- 👥 Voice cloning from user-provided audio samples
- 🚀 Built with LangChain for modular and scalable execution
- ⚡ Efficient inference for near real-time synthesis
- 🎛️ Customizable voice parameters (speed, voice selection, language)
- 🌐 Flexible API and CLI for easy interaction
- ☁️ RunPod serverless deployment for GPU acceleration

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd voice_ai_agent

# Install dependencies
pip install -r requirements.txt

# Install espeak-ng (required for some voice functionalities)
# For Ubuntu/Debian
sudo apt-get install espeak-ng

# For macOS
brew install espeak-ng

# For Windows
# Download and install from https://github.com/espeak-ng/espeak-ng/releases
```

## Usage

### Command Line Interface

#### Using Kokoro-82M (Default)

```bash
# Generate speech from text with Kokoro model
python cli.py --model kokoro --text "Hello, I am your voice assistant" --output output.wav

# Clone a voice from a sample audio with Kokoro
python cli.py --model kokoro --text "This is a cloned voice" --clone-from sample.wav --output cloned_output.wav

# Adjust voice parameters with Kokoro
python cli.py --model kokoro --text "Custom voice parameters" --voice af_heart --speed 1.2 --output custom_params.wav

# List available Kokoro voices
python cli.py --model kokoro --list-voices
```

#### Using Zonos-v0.1-hybrid

```bash
# Generate speech from text with Zonos model
python cli.py --model zonos --text "Hello, I am your voice assistant" --output zonos_output.wav

# Clone a voice from a sample audio with Zonos
python cli.py --model zonos --text "This is a cloned voice" --clone-from sample.wav --output zonos_cloned.wav

# Setting a default voice for Zonos
python cli.py --model zonos --clone-from sample.wav --make-default

# Specify language for Zonos (supports various languages)
python cli.py --model zonos --text "Hello, I am your voice assistant" --language en-us --output zonos_output.wav

# List available Zonos voices
python cli.py --model zonos --list-voices
```

### API

Start the API server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Then use the API endpoints:

- `/synthesize`: Generate speech from text with either Kokoro or Zonos model
- `/clone`: Clone a voice from audio sample with either model
- `/voices`: List available voices for the selected model
- `/synthesize-with-clone`: Clone voice and synthesize in one step

#### API Examples

##### Using Kokoro Model
```bash
# Synthesize text with Kokoro (default)
curl -X POST "http://localhost:8000/synthesize" \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello from Kokoro model", "model_type": "kokoro", "voice": "af_heart"}' \
    --output kokoro_speech.wav

# List Kokoro voices
curl "http://localhost:8000/voices?model_type=kokoro"
```

##### Using Zonos Model
```bash
# Synthesize text with Zonos
curl -X POST "http://localhost:8000/synthesize" \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello from Zonos model", "model_type": "zonos", "language": "en-us"}' \
    --output zonos_speech.wav

# Clone a voice with Zonos and set as default
curl -X POST "http://localhost:8000/clone" \
    -F "audio=@sample.wav" \
    -F "model_type=zonos" \
    -F "make_default=true"

# List Zonos voices
curl "http://localhost:8000/voices?model_type=zonos"
```

## RunPod Serverless Deployment

This project supports deployment on RunPod serverless for GPU-accelerated inference.

### Deployment Steps

1. Build the Docker container:
   ```bash
   docker build -t your-username/voice-ai-agent:latest .
   ```

2. Push to Docker Hub:
   ```bash
   docker push your-username/voice-ai-agent:latest
   ```

3. Create a new serverless endpoint on RunPod with your Docker image URL

### API Usage (RunPod)

Once deployed, you can make requests to your RunPod endpoint:

#### With Kokoro Model (default)
```bash
curl -X POST "https://api.runpod.ai/v2/{your-endpoint-id}/run" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "operation": "synthesize",
            "model_type": "kokoro",
            "text": "Hello from RunPod serverless!",
            "voice": "af_heart"
        }
    }'
```

#### With Zonos Model
```bash
curl -X POST "https://api.runpod.ai/v2/{your-endpoint-id}/run" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "operation": "synthesize",
            "model_type": "zonos",
            "text": "Hello from RunPod serverless!",
            "language": "en-us"
        }
    }'
```

**Operations:**
- `synthesize`: Generate speech from text (supports both models)
- `clone`: Clone a voice from audio URL (supports both models)
- `synthesize_with_clone`: Clone voice and synthesize in one step
- `list_voices`: List available voices for the selected model

For detailed deployment instructions, see [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md).

## Architecture

The Voice AI Agent is built using LangChain for workflow management and supports two state-of-the-art text-to-speech models:

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M): A lightweight yet expressive TTS model with support for multiple languages and preset voices.
  
- [Zonos-v0.1-hybrid](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid): An advanced TTS model with superior voice cloning capabilities and high-quality audio output.

The modular architecture allows for seamless switching between models and easy extension of functionality.

## Voice Cloning

The voice cloning functionality varies by model:

### Kokoro Voice Cloning

With Kokoro, voice cloning works by processing audio samples to extract voice characteristics using a simplified voice conversion approach. The system uses these characteristics to modify the output speech, maintaining a balance between voice similarity and speech quality.

### Zonos Voice Cloning

Zonos offers more advanced voice cloning with higher fidelity to the source voice. It creates speaker embeddings from reference audio that capture detailed voice characteristics. These embeddings can be saved and reused, allowing for consistent voice reproduction across multiple sessions. Users can also set a default voice for convenience.

## Testing

To test the RunPod handler locally before deployment:

```bash
python test_runpod_handler.py
```

This will verify that all operations work correctly with the RunPod handler.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
