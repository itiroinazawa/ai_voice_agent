#!/usr/bin/env python
"""
Command-line interface for the Voice AI Agent
"""

import argparse
import os
import sys
from pathlib import Path

from voice_agent.chain import create_voice_agent

def main():
    """
    Main entry point for the CLI
    """
    parser = argparse.ArgumentParser(description="Voice AI Agent - Generate high-quality speech from text")
    
    # Text input options
    parser.add_argument("--text", "-t", type=str, help="Text to synthesize")
    parser.add_argument("--file", "-f", type=str, help="Text file to synthesize")
    
    # Voice options
    parser.add_argument("--voice", type=str, default="af_heart", 
                        help="Voice to use (preset or cloned voice ID)")
    parser.add_argument("--clone-from", type=str, 
                        help="Audio file to clone voice from")
    parser.add_argument("--voice-id", type=str, 
                        help="ID for cloned voice (if not provided, a unique ID will be generated)")
    parser.add_argument("--list-voices", action="store_true", 
                        help="List available voices")
    
    # Output options
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output audio file path")
    
    # Voice parameters
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed factor (1.0 is normal)")
    
    args = parser.parse_args()
    
    # Create the voice agent
    agent = create_voice_agent()
    tts_engine = agent["tts_engine"]
    
    # List available voices
    if args.list_voices:
        voices = tts_engine.list_voices()
        print("Available voices:")
        print("  Preset voices:")
        for voice in voices["preset"]:
            print(f"    - {voice}")
        print("  Cloned voices:")
        for voice in voices["cloned"]:
            print(f"    - {voice}")
        return 0
    
    # Check if we need to clone a voice
    if args.clone_from:
        if not os.path.exists(args.clone_from):
            print(f"Error: Audio file {args.clone_from} does not exist")
            return 1
        
        print(f"Cloning voice from {args.clone_from}...")
        voice_id = agent["cloning_chain"]({
            "audio_path": args.clone_from,
            "voice_id": args.voice_id
        })["voice_id"]
        
        print(f"Voice cloned with ID: {voice_id}")
        
        # Use the cloned voice
        voice = voice_id
    else:
        voice = args.voice
    
    # Get text content
    if args.text:
        text = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: Text file {args.file} does not exist")
            return 1
        
        with open(args.file, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        print("Error: Please provide text using --text or --file")
        return 1
    
    # Synthesize speech
    print(f"Synthesizing speech with voice {voice}...")
    audio_path = agent["voice_chain"]({
        "text": text,
        "voice_id": voice,
        "speed": args.speed,
        "output_path": args.output
    })["audio_path"]
    
    print(f"Audio saved to {audio_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
