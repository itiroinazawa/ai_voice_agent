#!/usr/bin/env python
"""
Basic example of using the Voice AI Agent
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import voice_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_agent.chain import create_voice_agent

def main():
    """
    Run a basic example of the Voice AI Agent
    """
    print("Voice AI Agent - Basic Example")
    print("==============================")
    
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create the voice agent
    agent = create_voice_agent()
    
    # Example text to synthesize
    text = """
    Hello there! This is an example of the Voice AI Agent.
    I can generate high-quality speech from text.
    I can also clone voices from audio samples.
    """
    
    # Synthesize speech with default voice
    print("Synthesizing speech with default voice...")
    output_path = output_dir / "basic_example.wav"
    
    audio_path = agent["voice_chain"]({
        "text": text,
        "voice_id": "af_heart",
        "speed": 1.0,
        "output_path": str(output_path)
    })["audio_path"]
    
    print(f"Audio saved to {audio_path}")
    
    # List available voices
    voices = agent["tts_engine"].list_voices()
    print("\nAvailable voices:")
    print("  Preset voices:")
    for voice in voices["preset"]:
        print(f"    - {voice}")
    print("  Cloned voices:")
    for voice in voices["cloned"]:
        print(f"    - {voice}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
