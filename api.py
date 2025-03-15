#!/usr/bin/env python
"""
FastAPI implementation for the Voice AI Agent
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

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

# Create the voice agent
agent = create_voice_agent()
tts_engine = agent["tts_engine"]

# Create temp directory if it doesn't exist
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Pydantic models for request/response
class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0

class VoiceListResponse(BaseModel):
    preset: List[str]
    cloned: List[str]

class VoiceIDResponse(BaseModel):
    voice_id: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Voice AI Agent API is running"}

@app.get("/voices", response_model=VoiceListResponse)
async def list_voices():
    """List available voices"""
    return tts_engine.list_voices()

@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text
    """
    # Create temporary output file
    output_file = TEMP_DIR / f"{next(tempfile._get_candidate_names())}.wav"
    
    try:
        audio_path = agent["voice_chain"]({
            "text": request.text,
            "voice_id": request.voice,
            "speed": request.speed,
            "output_path": str(output_file)
        })["audio_path"]
        
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
    voice_id: Optional[str] = Form(None)
):
    """
    Clone a voice from an audio sample
    """
    # Save uploaded file
    temp_audio = TEMP_DIR / f"{next(tempfile._get_candidate_names())}{os.path.splitext(audio.filename)[1]}"
    
    try:
        with open(temp_audio, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        cloned_voice_id = agent["cloning_chain"]({
            "audio_path": str(temp_audio),
            "voice_id": voice_id
        })["voice_id"]
        
        return {"voice_id": cloned_voice_id}
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
    speed: float = Form(1.0)
):
    """
    Clone a voice and synthesize text with it in one step
    """
    # Save uploaded file
    temp_audio = TEMP_DIR / f"{next(tempfile._get_candidate_names())}{os.path.splitext(audio.filename)[1]}"
    output_file = TEMP_DIR / f"{next(tempfile._get_candidate_names())}.wav"
    
    try:
        with open(temp_audio, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Clone voice
        voice_id = agent["cloning_chain"]({
            "audio_path": str(temp_audio),
            "voice_id": None
        })["voice_id"]
        
        # Synthesize speech with cloned voice
        audio_path = agent["voice_chain"]({
            "text": text,
            "voice_id": voice_id,
            "speed": speed,
            "output_path": str(output_file)
        })["audio_path"]
        
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
