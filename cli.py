#!/usr/bin/env python
"""
Command-line interface for the Voice AI Agent
"""

import argparse
import os
import sys

from voice_agent.chain import create_voice_agent

def main():
    """
    Main entry point for the CLI
    """
    parser = argparse.ArgumentParser(description="Voice AI Agent - Generate high-quality speech from text")
    
    # Text input options
    parser.add_argument("--text", "-t", type=str, help="Text to synthesize")
    parser.add_argument("--file", "-f", type=str, help="Text file to synthesize")
    
    # Model selection
    parser.add_argument("--model", type=str, default="kokoro", choices=["kokoro", "zonos"],
                        help="TTS model to use ('kokoro' or 'zonos')")
    
    # Voice options
    parser.add_argument("--voice", type=str, default="af_heart", 
                        help="Voice to use (preset or cloned voice ID for Kokoro)")
    parser.add_argument("--clone-from", type=str, 
                        help="Audio file to clone voice from")
    parser.add_argument("--reference-audio", type=str,
                        help="Reference audio for Zonos voice cloning")
    parser.add_argument("--voice-id", type=str, 
                        help="ID for cloned voice (if not provided, a unique ID will be generated)")
    parser.add_argument("--make-default", action="store_true",
                        help="Set the cloned voice as default (Zonos only)")
    parser.add_argument("--list-voices", action="store_true", 
                        help="List available voices")
    parser.add_argument("--language", type=str, default="en-us",
                        help="Language code for Zonos (e.g., 'en-us', 'ja-jp')")
    
    # Output options
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output audio file path")
    
    # Voice parameters
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed factor (1.0 is normal)")
    
    args = parser.parse_args()
    
    # Create the voice agent with the selected model
    agent = create_voice_agent(model_type=args.model, language=args.language)
    tts_engine = agent["tts_engine"]
    
    # List available voices
    if args.list_voices:
        voices = tts_engine.list_voices()
        print(f"Available {args.model} voices:")
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
        
        print(f"Cloning {args.model} voice from {args.clone_from}...")
        voice_id = agent["cloning_chain"]({
            "audio_path": args.clone_from,
            "voice_id": args.voice_id,
            "make_default": args.make_default
        })["voice_id"]
        
        print(f"{args.model} voice cloned with ID: {voice_id}")
        
        # Use the cloned voice for Kokoro
        if args.model == 'kokoro':
            voice = voice_id
        else:
            # For Zonos, we'll use the reference audio directly
            args.reference_audio = args.clone_from
    
    # For Zonos, always use the reference audio if provided
    reference_audio = args.reference_audio if args.model == 'zonos' else None
    if args.model == 'zonos' and not reference_audio and not args.clone_from:
        print("Warning: Using Zonos without reference audio. Make sure you have a default voice set.")
    
    # Set voice for Kokoro
    if args.model == 'kokoro' and not 'voice_id' in locals():
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
    if args.model == 'kokoro':
        print(f"Synthesizing speech with Kokoro using voice {voice}...")
        audio_path = agent["voice_chain"]({
            "text": text,
            "voice_id": voice,
            "speed": args.speed,
            "output_path": args.output
        })["audio_path"]
    else:  # Zonos model
        print(f"Synthesizing speech with Zonos...")
        audio_path = agent["voice_chain"]({
            "text": text,
            "reference_audio": reference_audio,  # Can be None if using default voice
            "speed": args.speed,
            "output_path": args.output
        })["audio_path"]
    
    print(f"Audio saved to {audio_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
