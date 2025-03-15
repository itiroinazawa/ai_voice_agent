"""
Core TTS engine for the Voice AI Agent supporting multiple TTS models including:
- Kokoro-82M
- Zonos-v0.1-hybrid
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Dict

import torch
import numpy as np
import soundfile as sf
import torchaudio
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import TTS engines conditionally to avoid errors if one is not installed
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    logger.warning("Kokoro not available. Install with 'pip install kokoro'")
    KOKORO_AVAILABLE = False

try:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    ZONOS_AVAILABLE = True
except ImportError:
    logger.warning("Zonos not available. Install with 'pip install zonos'")
    ZONOS_AVAILABLE = False


class TTSEngine:
    """
    Text-to-Speech engine that supports multiple TTS models:
    - Kokoro-82M: Lightweight, fast TTS model
    - Zonos-v0.1-hybrid: Higher quality, more expressive TTS model
    """
    def __init__(self, 
                 model_type: str = 'kokoro',
                 lang_code: str = 'a',
                 language: str = 'en-us',
                 device: Optional[str] = None,
                 voices_dir: Optional[str] = None):
        """
        Initialize the TTS engine.
        
        Args:
            model_type: TTS model to use ('kokoro' or 'zonos')
            lang_code: Language code for Kokoro ('a' for American English, 'b' for British English, etc.)
            language: Language code for Zonos ('en-us', 'ja-jp', etc.)
            device: Device to run inference on ('cuda' or 'cpu')
            voices_dir: Directory to store voice embeddings for cloning
        """
        self.model_type = model_type.lower()
        self.lang_code = lang_code
        self.language = language
        
        # Set device (use CUDA if available)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing TTS Engine with {self.model_type} model on {self.device}")
        
        # Initialize appropriate model
        if self.model_type == 'kokoro':
            if not KOKORO_AVAILABLE:
                raise ImportError("Kokoro is not installed. Install with 'pip install kokoro'")
            
            try:
                self.pipeline = KPipeline(lang_code=lang_code)
                logger.info("Kokoro pipeline initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Kokoro pipeline: {e}")
                raise
                
            # Available preset voices for Kokoro
            self.available_voices = ['af_heart', 'af_woh', 'am_standard']
                
        elif self.model_type == 'zonos':
            if not ZONOS_AVAILABLE:
                raise ImportError("Zonos is not installed. Install with 'pip install zonos'")
                
            try:
                self.model = Zonos.from_pretrained(
                    "Zyphra/Zonos-v0.1-hybrid", 
                    device=self.device
                )
                logger.info("Zonos model initialized successfully")
                
                # Zonos doesn't have preset voices like Kokoro
                # Instead, it generates voices from reference audio
                self.available_voices = []
                
            except Exception as e:
                logger.error(f"Failed to initialize Zonos model: {e}")
                raise
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Choose 'kokoro' or 'zonos'")
        
        # Set up voices directory
        if voices_dir is None:
            self.voices_dir = Path("voices")
        else:
            self.voices_dir = Path(voices_dir)
        
        # Create voices directory if it doesn't exist
        if not self.voices_dir.exists():
            self.voices_dir.mkdir(parents=True, exist_ok=True)
    
    def synthesize(self, 
                  text: str, 
                  voice: str = 'af_heart',
                  speed: float = 1.0,
                  output_path: Optional[str] = None,
                  split_pattern: str = r'\n+',
                  reference_audio: Optional[str] = None) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use (preset or cloned voice ID for Kokoro, or ignored for Zonos)
            speed: Speed factor (1.0 is normal)
            output_path: Path to save the output audio
            split_pattern: Pattern to split text into chunks (for Kokoro)
            reference_audio: Path to reference audio for voice cloning (required for Zonos)
            
        Returns:
            Path to the generated audio file
        """
        logger.info(f"Synthesizing text with model {self.model_type}")
        
        # Create output path if not provided
        if output_path is None:
            # Create a temporary file if no output path is provided
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        if self.model_type == 'kokoro':
            return self._synthesize_kokoro(text, voice, speed, output_path, split_pattern)
        elif self.model_type == 'zonos':
            return self._synthesize_zonos(text, reference_audio, output_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _synthesize_kokoro(self,
                          text: str,
                          voice: str = 'af_heart',
                          speed: float = 1.0,
                          output_path: str = None,
                          split_pattern: str = r'\n+') -> str:
        """
        Synthesize speech using Kokoro-82M model.
        """
        logger.info(f"Synthesizing text with Kokoro using voice {voice}")
        
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
        sf.write(output_path, combined_audio, 24000)
        logger.info(f"Saved Kokoro synthesized audio to {output_path}")
        
        return output_path
    
    def _synthesize_zonos(self,
                        text: str,
                        reference_audio: Optional[str] = None,
                        output_path: str = None) -> str:
        """
        Synthesize speech using Zonos-v0.1-hybrid model.
        """
        logger.info(f"Synthesizing text with Zonos")
        
        if reference_audio is None:
            # Check if there's a default speaker embedding
            speaker_dir = self.voices_dir / "zonos_default"
            speaker_file = speaker_dir / "speaker_embedding.pt"
            
            if speaker_file.exists():
                logger.info(f"Using default speaker embedding from {speaker_file}")
                speaker = torch.load(speaker_file, map_location=self.device)
            else:
                raise ValueError("Reference audio is required for Zonos when no default speaker embedding exists")
        else:
            # Load reference audio and create speaker embedding
            logger.info(f"Creating speaker embedding from reference audio: {reference_audio}")
            wav, sampling_rate = torchaudio.load(reference_audio)
            if wav.shape[0] > 1:  # Convert stereo to mono if needed
                wav = torch.mean(wav, dim=0, keepdim=True)
            speaker = self.model.make_speaker_embedding(wav, sampling_rate)
            
            # Save the speaker embedding for future use
            speaker_dir = self.voices_dir / f"zonos_{Path(reference_audio).stem}"
            speaker_dir.mkdir(exist_ok=True, parents=True)
            speaker_file = speaker_dir / "speaker_embedding.pt"
            torch.save(speaker, speaker_file)
        
        # Create conditioning dictionary
        cond_dict = make_cond_dict(
            text=text, 
            speaker=speaker, 
            language=self.language
        )
        
        # Prepare conditioning and generate audio
        conditioning = self.model.prepare_conditioning(cond_dict)
        codes = self.model.generate(conditioning)
        wavs = self.model.autoencoder.decode(codes).cpu()
        
        # Save audio
        torchaudio.save(output_path, wavs[0], self.model.autoencoder.sampling_rate)
        logger.info(f"Saved Zonos synthesized audio to {output_path}")
        
        return output_path
    
    def clone_voice(self, audio_path: str, 
                   voice_id: Optional[str] = None,
                   make_default: bool = False) -> str:
        """
        Clone a voice from an audio sample.
        
        Args:
            audio_path: Path to the audio sample
            voice_id: ID for the cloned voice
            make_default: Whether to set this voice as the default for the model
            
        Returns:
            Voice ID for the cloned voice
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        if self.model_type == 'kokoro':
            return self._clone_voice_kokoro(audio_path, voice_id)
        elif self.model_type == 'zonos':
            return self._clone_voice_zonos(audio_path, voice_id, make_default)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _clone_voice_kokoro(self, audio_path: str, voice_id: Optional[str] = None) -> str:
        """
        Clone a voice using Kokoro's approach.
        """
        if voice_id is None:
            # Generate a unique voice ID if none is provided
            voice_id = f"kokoro_{os.path.basename(audio_path).split('.')[0]}"
        
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
        
        logger.info(f"Cloned Kokoro voice saved with ID: {voice_id}")
        return voice_id
    
    def _clone_voice_zonos(self, audio_path: str, voice_id: Optional[str] = None, make_default: bool = False) -> str:
        """
        Clone a voice using Zonos's speaker embedding approach.
        """
        if voice_id is None:
            # Generate a unique voice ID if none is provided
            voice_id = f"zonos_{os.path.basename(audio_path).split('.')[0]}"
        
        # Create speaker embedding
        logger.info(f"Creating Zonos speaker embedding from: {audio_path}")
        wav, sampling_rate = torchaudio.load(audio_path)
        if wav.shape[0] > 1:  # Convert stereo to mono if needed
            wav = torch.mean(wav, dim=0, keepdim=True)
        speaker_embedding = self.model.make_speaker_embedding(wav, sampling_rate)
        
        # Save the speaker embedding
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(exist_ok=True, parents=True)
        embedding_path = voice_dir / "speaker_embedding.pt"
        torch.save(speaker_embedding, embedding_path)
        
        # Save a copy of the original audio for reference
        sample = AudioSegment.from_file(audio_path)
        sample_path = voice_dir / "sample.wav"
        sample.export(sample_path, format="wav")
        
        # If this is set as default, create a link or copy to the default location
        if make_default:
            default_dir = self.voices_dir / "zonos_default"
            default_dir.mkdir(exist_ok=True, parents=True)
            default_path = default_dir / "speaker_embedding.pt"
            
            # Save a copy as the default embedding
            torch.save(speaker_embedding, default_path)
            logger.info(f"Set {voice_id} as default Zonos voice")
        
        logger.info(f"Cloned Zonos voice saved with ID: {voice_id}")
        return voice_id
    
    def list_voices(self) -> Dict[str, List[str]]:
        """
        List available voices for the current model.
        
        Returns:
            Dictionary with preset and cloned voices
        """
        if self.model_type == 'kokoro':
            return self._list_voices_kokoro()
        elif self.model_type == 'zonos':
            return self._list_voices_zonos()
        else:
            return {"preset": [], "cloned": []}
    
    def _list_voices_kokoro(self) -> Dict[str, List[str]]:
        """
        List available Kokoro voices.
        """
        # Get cloned Kokoro voices
        cloned_voices = []
        if self.voices_dir.exists():
            cloned_voices = [d.name for d in self.voices_dir.iterdir() 
                           if d.is_dir() and d.name.startswith("kokoro_")]
        
        return {
            "preset": self.available_voices,
            "cloned": cloned_voices
        }
    
    def _list_voices_zonos(self) -> Dict[str, List[str]]:
        """
        List available Zonos voices.
        """
        # Get cloned Zonos voices
        cloned_voices = []
        if self.voices_dir.exists():
            # Only include directories that have the speaker_embedding.pt file
            cloned_voices = [d.name for d in self.voices_dir.iterdir() 
                           if d.is_dir() and d.name.startswith("zonos_") 
                           and (d / "speaker_embedding.pt").exists()]
        
        # Check if we have a default voice
        has_default = (self.voices_dir / "zonos_default" / "speaker_embedding.pt").exists()
        
        return {
            "preset": ["default"] if has_default else [],
            "cloned": cloned_voices
        }
