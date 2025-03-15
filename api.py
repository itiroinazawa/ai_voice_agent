#!/usr/bin/env python
"""
FastAPI implementation for the Voice AI Agent
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from voice_agent.chain import create_voice_agent

# Initialize FastAPI app
app = FastAPI(
    title="Voice AI Agent API",
    description="API for text-to-speech synthesis and voice cloning",
    version="1.0.0",
)

# Initialize agents for both model types
kokoro_agent = create_voice_agent(model_type="kokoro")
zonos_agent = create_voice_agent(model_type="zonos")

# Default to Kokoro for backward compatibility
active_agent = kokoro_agent

# Get the engines for easy access
kokoro_tts_engine = kokoro_agent["tts_engine"]
zonos_tts_engine = zonos_agent["tts_engine"]

# Create temp directory if it doesn't exist
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Pydantic models for request/response
class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "af_heart"  # Used for Kokoro model
    reference_audio: Optional[str] = None  # Path to reference audio for Zonos
    model_type: str = "kokoro"  # 'kokoro' or 'zonos'
    language: str = "en-us"  # Language code (primarily for Zonos)
    speed: float = 1.0

class VoiceListResponse(BaseModel):
    preset: List[str]
    cloned: List[str]
    model_type: str

class VoiceIDResponse(BaseModel):
    voice_id: str
    model_type: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Voice AI Agent API is running"}

@app.get("/voices", response_model=VoiceListResponse)
async def list_voices(model_type: str = "kokoro"):
    """List available voices for the specified model"""
    if model_type == "kokoro":
        voices = kokoro_tts_engine.list_voices()
    elif model_type == "zonos":
        voices = zonos_tts_engine.list_voices()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
    
    # Add the model type to the response
    return {**voices, "model_type": model_type}

@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text using either Kokoro or Zonos model
    """
    # Create temporary output file
    output_file = TEMP_DIR / f"{next(tempfile._get_candidate_names())}.wav"
    
    try:
        # Select the appropriate agent based on the model type
        if request.model_type == "kokoro":
            current_agent = kokoro_agent
            audio_path = current_agent["voice_chain"]({
                "text": request.text,
                "voice_id": request.voice,
                "speed": request.speed,
                "output_path": str(output_file)
            })["audio_path"]
        elif request.model_type == "zonos":
            current_agent = zonos_agent
            
            # For Zonos, use reference_audio if provided
            params = {
                "text": request.text,
                "speed": request.speed,
                "output_path": str(output_file)
            }
            
            # Add reference_audio if provided
            if request.reference_audio:
                params["reference_audio"] = request.reference_audio
            
            audio_path = current_agent["voice_chain"](params)["audio_path"]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {request.model_type}")
        
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=os.path.basename(audio_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clone", response_model=VoiceIDResponse)
async def clone_voice(
    audio: UploadFile = File(...),
    voice_id: Optional[str] = Form(None),
    model_type: str = Form("kokoro"),
    make_default: bool = Form(False),
    language: str = Form("en-us")
):
    """
    Clone a voice from an audio sample for either Kokoro or Zonos model
    """
    # Save uploaded file
    temp_audio = TEMP_DIR / f"{next(tempfile._get_candidate_names())}{os.path.splitext(audio.filename)[1]}"
    
    try:
        with open(temp_audio, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Select the appropriate agent based on the model type
        if model_type == "kokoro":
            current_agent = kokoro_agent
        elif model_type == "zonos":
            current_agent = zonos_agent
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
        
        cloned_voice_id = current_agent["cloning_chain"]({
            "audio_path": str(temp_audio),
            "voice_id": voice_id,
            "make_default": make_default
        })["voice_id"]
        
        return {"voice_id": cloned_voice_id, "model_type": model_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if temp_audio.exists():
            temp_audio.unlink()

@app.post("/synthesize-with-clone")
async def synthesize_with_clone(
    text: str = Form(...),
    audio: UploadFile = File(...),
    speed: float = Form(1.0),
    model_type: str = Form("kokoro"),
    make_default: bool = Form(False),
    language: str = Form("en-us")
):
    """
    Clone a voice and synthesize text with it in one step, supporting both Kokoro and Zonos models
    """
    # Save uploaded file
    temp_audio = TEMP_DIR / f"{next(tempfile._get_candidate_names())}{os.path.splitext(audio.filename)[1]}"
    output_file = TEMP_DIR / f"{next(tempfile._get_candidate_names())}.wav"
    
    try:
        with open(temp_audio, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Select the appropriate agent based on the model type
        if model_type == "kokoro":
            current_agent = kokoro_agent
            
            # Clone voice with Kokoro
            voice_id = current_agent["cloning_chain"]({
                "audio_path": str(temp_audio),
                "voice_id": None
            })["voice_id"]
            
            # Synthesize speech with cloned voice using Kokoro
            audio_path = current_agent["voice_chain"]({
                "text": text,
                "voice_id": voice_id,
                "speed": speed,
                "output_path": str(output_file)
            })["audio_path"]
            
        elif model_type == "zonos":
            current_agent = zonos_agent
            
            # Clone voice with Zonos
            voice_id = current_agent["cloning_chain"]({
                "audio_path": str(temp_audio),
                "voice_id": None,
                "make_default": make_default
            })["voice_id"]
            
            # For Zonos, use the reference audio directly
            audio_path = current_agent["voice_chain"]({
                "text": text,
                "reference_audio": str(temp_audio),  # Use the uploaded audio file directly
                "speed": speed,
                "output_path": str(output_file)
            })["audio_path"]
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
        
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=os.path.basename(audio_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if temp_audio.exists():
            temp_audio.unlink()

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
