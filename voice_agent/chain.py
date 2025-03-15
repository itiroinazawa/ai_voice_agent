"""
LangChain integration for the Voice AI Agent
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.chains import Chain

from voice_agent.tts_engine import TTSEngine

class VoiceChain(Chain):
    """
    LangChain implementation for the Voice AI Agent that processes text and
    generates speech using the TTS engine.
    """
    
    tts_engine: TTSEngine
    output_key: str = "audio_path"
    
    @property
    def input_keys(self) -> List[str]:
        """Return input keys."""
        return ["text", "voice_id", "speed", "output_path"]
    
    @property
    def output_keys(self) -> List[str]:
        """Return output keys."""
        return [self.output_key]
    
    def _call(
        self, 
        inputs: Dict[str, Any], 
        run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, Any]:
        """
        Process the inputs and generate speech.
        
        Args:
            inputs: Input dictionary with text, voice, etc.
            run_manager: Callback manager for the chain run
            
        Returns:
            Dictionary with path to the generated audio file
        """
        # Extract inputs
        text = inputs.get("text", "")
        voice_id = inputs.get("voice_id", "af_heart")
        speed = float(inputs.get("speed", 1.0))
        output_path = inputs.get("output_path")
        
        # Synthesize speech
        audio_path = self.tts_engine.synthesize(
            text=text,
            voice=voice_id,
            speed=speed,
            output_path=output_path
        )
        
        return {self.output_key: audio_path}

class VoiceCloningChain(Chain):
    """
    LangChain implementation for voice cloning functionality.
    """
    
    tts_engine: TTSEngine
    output_key: str = "voice_id"
    
    @property
    def input_keys(self) -> List[str]:
        """Return input keys."""
        return ["audio_path", "voice_id"]
    
    @property
    def output_keys(self) -> List[str]:
        """Return output keys."""
        return [self.output_key]
    
    def _call(
        self, 
        inputs: Dict[str, Any], 
        run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, Any]:
        """
        Clone a voice from an audio sample.
        
        Args:
            inputs: Input dictionary with audio path and optional voice ID
            run_manager: Callback manager for the chain run
            
        Returns:
            Dictionary with the voice ID for the cloned voice
        """
        # Extract inputs
        audio_path = inputs.get("audio_path")
        voice_id = inputs.get("voice_id")
        
        # Clone voice
        cloned_voice_id = self.tts_engine.clone_voice(
            audio_path=audio_path,
            voice_id=voice_id
        )
        
        return {self.output_key: cloned_voice_id}

def create_voice_agent():
    """
    Create and configure the voice agent with LangChain components.
    
    Returns:
        Dictionary with voice and cloning chains
    """
    # Initialize TTS engine
    tts_engine = TTSEngine()
    
    # Create voice synthesis chain
    voice_chain = VoiceChain(tts_engine=tts_engine)
    
    # Create voice cloning chain
    cloning_chain = VoiceCloningChain(tts_engine=tts_engine)
    
    return {
        "voice_chain": voice_chain,
        "cloning_chain": cloning_chain,
        "tts_engine": tts_engine
    }
