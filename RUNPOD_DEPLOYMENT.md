# Deploying Voice AI Agent on RunPod Serverless

This guide walks you through deploying the Voice AI Agent on RunPod Serverless for efficient GPU-accelerated inference.

## Prerequisites

- A RunPod account with billing set up
- Docker installed on your local machine (for building and testing the container)
- Git to clone this repository

## Deployment Steps

### 1. Build the Docker Container

```bash
# Clone the repository if you haven't already
git clone <repository-url>
cd voice_ai_agent

# Build the Docker image
docker build -t your-dockerhub-username/voice-ai-agent:latest .
```

### 2. Test the Container Locally (Optional)

```bash
# Run the container locally to test
docker run --gpus all -p 8000:8000 your-dockerhub-username/voice-ai-agent:latest
```

### 3. Push the Container to Docker Hub

```bash
# Log in to Docker Hub
docker login

# Push the image
docker push your-dockerhub-username/voice-ai-agent:latest
```

### 4. Deploy on RunPod Serverless

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click on "New Endpoint"
3. Select "Custom Template" and enter your Docker image URL (`your-dockerhub-username/voice-ai-agent:latest`)
4. Configure your endpoint:
   - Name: `voice-ai-agent`
   - GPU Type: Select an appropriate GPU (e.g., NVIDIA T4 or better)
   - Min Handlers: 0 (scales down to zero when not in use)
   - Max Handlers: Set based on your expected load
   - Handler size: 1 (per handler GPU count)
   - Idle Timeout: 5-10 minutes recommended
5. Click "Deploy"

## API Usage

Once deployed, you can interact with your voice AI agent using the RunPod API. Below are examples using curl:

### Speech Synthesis

```bash
curl -X POST "https://api.runpod.ai/v2/{your-endpoint-id}/run" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "operation": "synthesize",
            "text": "Hello, this is a test of the voice AI agent on RunPod.",
            "voice": "af_heart",
            "speed": 1.0
        }
    }'
```

### Voice Cloning

```bash
curl -X POST "https://api.runpod.ai/v2/{your-endpoint-id}/run" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "operation": "clone",
            "audio_url": "https://example.com/sample.wav",
            "voice_id": "my_custom_voice"
        }
    }'
```

### Synthesis with Voice Cloning

```bash
curl -X POST "https://api.runpod.ai/v2/{your-endpoint-id}/run" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "operation": "synthesize_with_clone",
            "text": "This is spoken in the cloned voice.",
            "audio_url": "https://example.com/sample.wav",
            "speed": 1.0
        }
    }'
```

### List Available Voices

```bash
curl -X POST "https://api.runpod.ai/v2/{your-endpoint-id}/run" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "operation": "list_voices"
        }
    }'
```

## Response Format

Responses will be in JSON format, containing:

### For Synthesis Operations:
```json
{
    "audio_base64": "base64_encoded_audio_data",
    "content_type": "audio/wav",
    "duration": 3.25,
    "sample_rate": 24000
}
```

### For Voice Cloning:
```json
{
    "voice_id": "cloned_voice_identifier"
}
```

### For List Voices:
```json
{
    "preset": ["af_heart", "af_woh", "am_standard"],
    "cloned": ["my_custom_voice"]
}
```

## Performance Considerations

- For larger text inputs, consider breaking them into smaller chunks to avoid timeouts
- The first request may be slower due to cold start times (model loading)
- Adjust your RunPod handler settings based on actual usage patterns

## Monitoring and Cost Management

- Monitor your endpoint usage through the RunPod dashboard
- Set up budget alerts to avoid unexpected charges
- Adjust min/max handlers and idle timeout to optimize cost vs. performance
