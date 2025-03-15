# Voice AI Agent

An AI agent that generates high-quality audio output from text prompts with voice cloning capabilities. Supports deployment on RunPod serverless for GPU-accelerated inference.

## Features

- 🔊 Natural and expressive speech synthesis using Kokoro-82M
- 👥 Voice cloning from user-provided audio samples
- 🚀 Built with LangChain for modular and scalable execution
- ⚡ Efficient inference for near real-time synthesis
- 🎛️ Customizable voice parameters (speed, voice selection)
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

```bash
# Generate speech from text
python cli.py --text "Hello, I am your voice assistant" --output output.wav

# Clone a voice from a sample audio
python cli.py --text "This is a cloned voice" --clone-from sample.wav --output cloned_output.wav

# Adjust voice parameters
python cli.py --text "Custom voice parameters" --speed 1.2 --output custom_params.wav

# List available voices
python cli.py --list-voices
```

### API

Start the API server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Then use the API endpoints:

- `/synthesize`: Generate speech from text
- `/clone`: Clone a voice and generate speech
- `/voices`: List available voice presets
- `/synthesize-with-clone`: Clone voice and synthesize in one step

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

```bash
curl -X POST "https://api.runpod.ai/v2/{your-endpoint-id}/run" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "operation": "synthesize",
            "text": "Hello from RunPod serverless!"
        }
    }'
```

**Operations:**
- `synthesize`: Generate speech from text
- `clone`: Clone a voice from audio URL
- `synthesize_with_clone`: Clone voice and synthesize in one step
- `list_voices`: List available voices

For detailed deployment instructions, see [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md).

## Architecture

The Voice AI Agent is built using LangChain for workflow management and [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) for high-quality text-to-speech synthesis. The modular architecture allows for easy extension and customization.

## Voice Cloning

The voice cloning functionality allows the agent to replicate specific voices from provided audio samples. The system extracts voice characteristics from the sample and applies them to new synthesized speech.

## Testing

To test the RunPod handler locally before deployment:

```bash
python test_runpod_handler.py
```

This will verify that all operations work correctly with the RunPod handler.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
