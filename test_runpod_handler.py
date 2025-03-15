#!/usr/bin/env python
"""
Test script for the RunPod serverless handler
This simulates RunPod serverless job requests to test functionality locally
"""

import os
import json
import base64
import argparse
from pathlib import Path
from pydub import AudioSegment

# Import the handler from runpod_handler.py
from runpod_handler import handler, initialize_agent

def save_base64_audio(base64_str, output_path):
    """Save base64 encoded audio to a file"""
    audio_data = base64.b64decode(base64_str)
    with open(output_path, "wb") as f:
        f.write(audio_data)
    print(f"Saved audio to: {output_path}")

def test_synthesize():
    """Test the synthesize operation"""
    print("\n=== Testing Speech Synthesis ===")
    job = {
        "input": {
            "operation": "synthesize",
            "text": "This is a test of the speech synthesis functionality on RunPod.",
            "voice": "af_heart",
            "speed": 1.0
        }
    }
    
    result = handler(job)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return False
    
    print(f"Success! Received audio with duration: {result['duration']:.2f} seconds")
    print(f"Sample rate: {result['sample_rate']} Hz")
    
    # Save the audio to a file
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_synthesis.wav"
    save_base64_audio(result["audio_base64"], output_path)
    
    return True

def test_list_voices():
    """Test the list_voices operation"""
    print("\n=== Testing List Voices ===")
    job = {
        "input": {
            "operation": "list_voices"
        }
    }
    
    result = handler(job)
    print("Available voices:")
    print(f"Preset voices: {', '.join(result['preset'])}")
    print(f"Cloned voices: {', '.join(result['cloned'])}")
    
    return True

def test_voice_cloning():
    """Test the voice cloning operation"""
    print("\n=== Testing Voice Cloning ===")
    
    # First need to have a sample audio file
    # For testing, we'll use the synthesized audio from the first test
    test_output_dir = Path("test_output")
    sample_audio = test_output_dir / "test_synthesis.wav"
    
    if not sample_audio.exists():
        print(f"Sample audio file {sample_audio} not found. Run test_synthesize first.")
        return False
    
    # Since RunPod serverless expects a URL, we'll mock this part
    # In a real scenario, this would be a publicly accessible URL
    # Here we just pass the local path and mock the download function
    
    # For testing only, we'll monkey patch the download_files_from_urls function
    # This normally wouldn't be necessary in the real RunPod environment
    import runpod_handler
    orig_download_fn = runpod_handler.download_files_from_urls
    
    def mock_download_fn(urls):
        print(f"Mocking download from {urls}")
        return [str(sample_audio)]
    
    runpod_handler.download_files_from_urls = mock_download_fn
    
    job = {
        "input": {
            "operation": "clone",
            "audio_url": "mock://sample.wav",
            "voice_id": "test_cloned_voice"
        }
    }
    
    try:
        result = handler(job)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return False
        
        print(f"Success! Cloned voice with ID: {result['voice_id']}")
        
        # List voices again to verify the cloned voice appears
        list_job = {
            "input": {
                "operation": "list_voices"
            }
        }
        voices_result = handler(list_job)
        print("Updated voice list:")
        print(f"Preset voices: {', '.join(voices_result['preset'])}")
        print(f"Cloned voices: {', '.join(voices_result['cloned'])}")
        
        return True
    finally:
        # Restore the original function
        runpod_handler.download_files_from_urls = orig_download_fn

def test_synthesize_with_clone():
    """Test the synthesize_with_clone operation"""
    print("\n=== Testing Synthesize with Clone ===")
    
    # First need to have a sample audio file
    test_output_dir = Path("test_output")
    sample_audio = test_output_dir / "test_synthesis.wav"
    
    if not sample_audio.exists():
        print(f"Sample audio file {sample_audio} not found. Run test_synthesize first.")
        return False
    
    # Mock the download function
    import runpod_handler
    orig_download_fn = runpod_handler.download_files_from_urls
    
    def mock_download_fn(urls):
        print(f"Mocking download from {urls}")
        return [str(sample_audio)]
    
    runpod_handler.download_files_from_urls = mock_download_fn
    
    job = {
        "input": {
            "operation": "synthesize_with_clone",
            "text": "This is a test of the voice cloning and synthesis in one step.",
            "audio_url": "mock://sample.wav",
            "speed": 1.1
        }
    }
    
    try:
        result = handler(job)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return False
        
        print(f"Success! Synthesized with cloned voice ID: {result['voice_id']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Sample rate: {result['sample_rate']} Hz")
        
        # Save the audio to a file
        output_path = test_output_dir / "test_cloned_synthesis.wav"
        save_base64_audio(result["audio_base64"], output_path)
        
        return True
    finally:
        # Restore the original function
        runpod_handler.download_files_from_urls = orig_download_fn

def main():
    parser = argparse.ArgumentParser(description="Test the RunPod serverless handler")
    parser.add_argument("--skip-init", action="store_true", help="Skip initialization (use if already initialized)")
    args = parser.parse_args()
    
    print("=== RunPod Handler Testing ===")
    
    # Initialize the agent if not already done
    if not args.skip_init:
        print("Initializing Voice AI Agent... (this may take a moment)")
        initialize_agent()
    
    # Create test output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Run tests
    tests = [
        test_synthesize,
        test_list_voices,
        test_voice_cloning,
        test_synthesize_with_clone
    ]
    
    success_count = 0
    for test_fn in tests:
        if test_fn():
            success_count += 1
    
    print(f"\n=== Test Results: {success_count}/{len(tests)} tests passed ===")

if __name__ == "__main__":
    main()
