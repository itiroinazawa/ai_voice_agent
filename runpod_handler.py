#!/usr/bin/env python
"""
RunPod serverless handler for Voice AI Agent
"""

import os
import base64
import tempfile
import time
from pathlib import Path
import runpod
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
import soundfile as sf

from voice_agent.chain import create_voice_agent

# Global variables
AGENT = None
TEMP_DIR = Path("/tmp/voice_ai")
TEMP_DIR.mkdir(exist_ok=True, parents=True)

def initialize_agent():
    """Initialize the voice agent once when the container starts"""
    global AGENT
    print("Initializing Voice AI Agent...")
    start_time = time.time()
    AGENT = create_voice_agent()
    print(f"Voice AI Agent initialized in {time.time() - start_time:.2f} seconds")
    return AGENT

def audio_to_base64(audio_path):
    """Convert audio file to base64 string"""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

def handle_synthesize(job_input):
    """Handle speech synthesis request"""
    text = job_input.get("text", "")
    voice = job_input.get("voice", "af_heart")
    speed = float(job_input.get("speed", 1.0))
    
    if not text:
        return {"error": "No text provided for synthesis"}
    
    # Create output file path
    output_path = str(TEMP_DIR / f"{next(tempfile._get_candidate_names())}.wav")
    
    try:
        audio_path = AGENT["voice_chain"]({
            "text": text,
            "voice_id": voice,
            "speed": speed,
            "output_path": output_path
        })["audio_path"]
        
        # Convert audio to base64
        audio_base64 = audio_to_base64(audio_path)
        
        # Get audio information
        audio_data, sample_rate = sf.read(audio_path)
        duration = len(audio_data) / sample_rate
        
        return {
            "audio_base64": audio_base64,
            "content_type": "audio/wav",
            "duration": duration,
            "sample_rate": sample_rate
        }
    except Exception as e:
        return {"error": f"Synthesis failed: {str(e)}"}
    finally:
        # Clean up the temporary file
        if os.path.exists(output_path):
            os.remove(output_path)

def handle_clone(job_input):
    """Handle voice cloning request"""
    audio_url = job_input.get("audio_url")
    voice_id = job_input.get("voice_id")
    
    if not audio_url:
        return {"error": "No audio URL provided for voice cloning"}
    
    try:
        # Download the audio file
        downloaded_files = download_files_from_urls([audio_url])
        audio_path = downloaded_files[0]
        
        # Clone the voice
        cloned_voice_id = AGENT["cloning_chain"]({
            "audio_path": audio_path,
            "voice_id": voice_id
        })["voice_id"]
        
        return {
            "voice_id": cloned_voice_id
        }
    except Exception as e:
        return {"error": f"Voice cloning failed: {str(e)}"}
    finally:
        # Clean up downloaded files
        rp_cleanup()

def handle_synthesize_with_clone(job_input):
    """Handle combined voice cloning and synthesis request"""
    text = job_input.get("text", "")
    audio_url = job_input.get("audio_url")
    speed = float(job_input.get("speed", 1.0))
    
    if not text:
        return {"error": "No text provided for synthesis"}
    if not audio_url:
        return {"error": "No audio URL provided for voice cloning"}
    
    try:
        # Download the audio file
        downloaded_files = download_files_from_urls([audio_url])
        audio_path = downloaded_files[0]
        
        # Clone the voice
        voice_id = AGENT["cloning_chain"]({
            "audio_path": audio_path,
            "voice_id": None
        })["voice_id"]
        
        # Create output file path
        output_path = str(TEMP_DIR / f"{next(tempfile._get_candidate_names())}.wav")
        
        # Synthesize speech with cloned voice
        audio_path = AGENT["voice_chain"]({
            "text": text,
            "voice_id": voice_id,
            "speed": speed,
            "output_path": output_path
        })["audio_path"]
        
        # Convert audio to base64
        audio_base64 = audio_to_base64(audio_path)
        
        # Get audio information
        audio_data, sample_rate = sf.read(audio_path)
        duration = len(audio_data) / sample_rate
        
        return {
            "audio_base64": audio_base64,
            "content_type": "audio/wav",
            "duration": duration,
            "sample_rate": sample_rate,
            "voice_id": voice_id
        }
    except Exception as e:
        return {"error": f"Operation failed: {str(e)}"}
    finally:
        # Clean up downloaded files and temporary output
        rp_cleanup()
        if os.path.exists(output_path):
            os.remove(output_path)

def list_voices():
    """List available voices"""
    return AGENT["tts_engine"].list_voices()

def handler(job):
    """
    Handler function for RunPod serverless
    
    Expected input format:
    {
        "input": {
            "operation": "synthesize|clone|synthesize_with_clone|list_voices",
            "text": "Text to synthesize",
            "voice": "Voice ID to use",
            "audio_url": "URL to audio for cloning",
            "speed": 1.0
        }
    }
    """
    job_input = job["input"]
    operation = job_input.get("operation", "synthesize")
    
    if operation == "synthesize":
        return handle_synthesize(job_input)
    elif operation == "clone":
        return handle_clone(job_input)
    elif operation == "synthesize_with_clone":
        return handle_synthesize_with_clone(job_input)
    elif operation == "list_voices":
        return list_voices()
    else:
        return {"error": f"Unknown operation: {operation}"}

# Initialize the model when the container starts
initialize_agent()

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
