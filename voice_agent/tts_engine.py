"""
Core TTS engine for the Voice AI Agent using Kokoro-82M model.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Generator

import torch
import numpy as np
import soundfile as sf
from kokoro import KPipeline
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSEngine:
    """
    Text-to-Speech engine that uses Kokoro-82M for speech synthesis.
    """
    def __init__(self, 
                 lang_code: str = 'a',
                 device: Optional[str] = None,
                 voices_dir: Optional[str] = None):
        """
        Initialize the TTS engine.
        
        Args:
            lang_code: Language code ('a' for American English, 'b' for British English, etc.)
            device: Device to run inference on ('cuda' or 'cpu')
            voices_dir: Directory to store voice embeddings for cloning
        """
        self.lang_code = lang_code
        
        # Set device (use CUDA if available)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing TTS Engine on {self.device}")
        
        # Initialize Kokoro pipeline
        try:
            self.pipeline = KPipeline(lang_code=lang_code)
            logger.info("Kokoro pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro pipeline: {e}")
            raise
        
        # Set up voices directory
        if voices_dir is None:
            self.voices_dir = Path("voices")
        else:
            self.voices_dir = Path(voices_dir)
        
        # Create voices directory if it doesn't exist
        if not self.voices_dir.exists():
            self.voices_dir.mkdir(parents=True, exist_ok=True)
        
        # Available preset voices
        self.available_voices = ['af_heart', 'af_woh', 'am_standard']
    
    def synthesize(self, 
                  text: str, 
                  voice: str = 'af_heart',
                  speed: float = 1.0,
                  output_path: Optional[str] = None,
                  split_pattern: str = r'\n+') -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use (preset or cloned voice ID)
            speed: Speed factor (1.0 is normal)
            output_path: Path to save the output audio
            split_pattern: Pattern to split text into chunks
            
        Returns:
            Path to the generated audio file
        """
        logger.info(f"Synthesizing text with voice {voice}")
        
        # Generate audio segments
        audio_segments = []
        generator = self.pipeline(
            text, 
            voice=voice, 
            speed=speed, 
            split_pattern=split_pattern
        )
        
        for i, (gs, ps, audio) in enumerate(generator):
            logger.debug(f"Generated segment {i}: {gs[:30]}...")
            audio_segments.append(audio)
        
        # Combine audio segments
        combined_audio = np.concatenate(audio_segments)
        
        # Save audio to file
        if output_path is None:
            # Create a temporary file if no output path is provided
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        sf.write(output_path, combined_audio, 24000)
        logger.info(f"Saved synthesized audio to {output_path}")
        
        return output_path
    
    def clone_voice(self, audio_path: str, 
                   voice_id: Optional[str] = None) -> str:
        """
        Clone a voice from an audio sample.
        
        Args:
            audio_path: Path to the audio sample
            voice_id: ID for the cloned voice
            
        Returns:
            Voice ID for the cloned voice
        """
        # For the prototype, we'll implement a simple voice cloning approach
        # In a full implementation, this would involve extracting voice embeddings
        # and using them for synthesis
        
        if voice_id is None:
            # Generate a unique voice ID if none is provided
            voice_id = f"cloned_{os.path.basename(audio_path).split('.')[0]}"
        
        # Copy the audio sample to the voices directory
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(exist_ok=True)
        
        # Process the sample (convert to proper format if needed)
        sample = AudioSegment.from_file(audio_path)
        sample = sample.set_frame_rate(24000)
        sample = sample.set_channels(1)
        
        # Save processed sample
        sample_path = voice_dir / "sample.wav"
        sample.export(sample_path, format="wav")
        
        logger.info(f"Cloned voice saved with ID: {voice_id}")
        return voice_id
    
    def list_voices(self) -> Dict[str, List[str]]:
        """
        List available voices.
        
        Returns:
            Dictionary with preset and cloned voices
        """
        # Get cloned voices
        cloned_voices = []
        if self.voices_dir.exists():
            cloned_voices = [d.name for d in self.voices_dir.iterdir() if d.is_dir()]
        
        return {
            "preset": self.available_voices,
            "cloned": cloned_voices
        }
