"""
Main TTS Engine for SYSPIN Multi-lingual TTS
Loads and runs VITS models for inference
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
import numpy as np
import torch
from dataclasses import dataclass

from .config import LANGUAGE_CONFIGS, LanguageConfig, MODELS_DIR
from .tokenizer import TTSTokenizer, CharactersConfig, TextNormalizer
from .downloader import ModelDownloader

logger = logging.getLogger(__name__)


@dataclass
class TTSOutput:
    """Output from TTS synthesis"""

    audio: np.ndarray
    sample_rate: int
    duration: float
    voice: str
    text: str


class TTSEngine:
    """
    Multi-lingual TTS Engine using SYSPIN VITS models

    Supports 10 Indian languages with male/female voices:
    - Hindi, Bengali, Marathi, Telugu, Kannada
    - Bhojpuri, Chhattisgarhi, Maithili, Magahi, English
    """

    def __init__(
        self,
        models_dir: str = MODELS_DIR,
        device: str = "auto",
        preload_voices: Optional[List[str]] = None,
    ):
        """
        Initialize TTS Engine

        Args:
            models_dir: Directory containing downloaded models
            device: Device to run inference on ('cpu', 'cuda', 'mps', or 'auto')
            preload_voices: List of voice keys to preload into memory
        """
        self.models_dir = Path(models_dir)
        self.device = self._get_device(device)

        # Model cache
        self._models: Dict[str, torch.jit.ScriptModule] = {}
        self._tokenizers: Dict[str, TTSTokenizer] = {}

        # Downloader
        self.downloader = ModelDownloader(models_dir)

        # Text normalizer
        self.normalizer = TextNormalizer()

        # Preload specified voices
        if preload_voices:
            for voice in preload_voices:
                self.load_voice(voice)

        logger.info(f"TTS Engine initialized on device: {self.device}")

    def _get_device(self, device: str) -> torch.device:
        """Determine the best device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            # MPS has compatibility issues with some TorchScript models
            # Using CPU for now - still fast on Apple Silicon
            # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            #     return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def load_voice(self, voice_key: str, download_if_missing: bool = True) -> bool:
        """
        Load a voice model into memory

        Args:
            voice_key: Key from LANGUAGE_CONFIGS (e.g., 'hi_male')
            download_if_missing: Download model if not found locally

        Returns:
            True if loaded successfully
        """
        if voice_key in self._models:
            return True

        if voice_key not in LANGUAGE_CONFIGS:
            raise ValueError(f"Unknown voice: {voice_key}")

        config = LANGUAGE_CONFIGS[voice_key]
        model_dir = self.models_dir / voice_key

        # Check if model exists, download if needed
        model_path = model_dir / config.model_filename
        if not model_path.exists():
            if download_if_missing:
                logger.info(f"Model not found, downloading {voice_key}...")
                self.downloader.download_model(voice_key)
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")

        # Find the actual model file (names vary)
        model_files = list(model_dir.glob("*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No .pt file found in {model_dir}")

        actual_model_path = model_files[0]

        # Load tokenizer
        chars_path = model_dir / "chars.txt"
        if chars_path.exists():
            tokenizer = TTSTokenizer.from_chars_file(str(chars_path))
        else:
            # Try to find chars file
            chars_files = list(model_dir.glob("*chars*.txt"))
            if chars_files:
                tokenizer = TTSTokenizer.from_chars_file(str(chars_files[0]))
            else:
                raise FileNotFoundError(f"No chars.txt found in {model_dir}")

        # Load model
        logger.info(f"Loading model from {actual_model_path}")
        model = torch.jit.load(str(actual_model_path), map_location=self.device)
        model.eval()

        # Cache model and tokenizer
        self._models[voice_key] = model
        self._tokenizers[voice_key] = tokenizer

        logger.info(f"Loaded voice: {voice_key}")
        return True

    def unload_voice(self, voice_key: str):
        """Unload a voice to free memory"""
        if voice_key in self._models:
            del self._models[voice_key]
            del self._tokenizers[voice_key]
            torch.cuda.empty_cache() if self.device.type == "cuda" else None
            logger.info(f"Unloaded voice: {voice_key}")

    def synthesize(
        self,
        text: str,
        voice: str = "hi_male",
        speed: float = 1.0,
        normalize_text: bool = True,
    ) -> TTSOutput:
        """
        Synthesize speech from text

        Args:
            text: Input text to synthesize
            voice: Voice key (e.g., 'hi_male', 'bn_female')
            speed: Speech speed multiplier (0.5-2.0)
            normalize_text: Whether to apply text normalization

        Returns:
            TTSOutput with audio array and metadata
        """
        # Load voice if not cached
        if voice not in self._models:
            self.load_voice(voice)

        model = self._models[voice]
        tokenizer = self._tokenizers[voice]
        config = LANGUAGE_CONFIGS[voice]

        # Normalize text
        if normalize_text:
            text = self.normalizer.clean_text(text, config.code)

        # Tokenize
        token_ids = tokenizer.text_to_ids(text)
        x = torch.from_numpy(np.array(token_ids)).unsqueeze(0).to(self.device)

        # Generate audio
        with torch.no_grad():
            audio = model(x)

        # Convert to numpy
        audio_np = audio.squeeze().cpu().numpy()

        # Apply speed adjustment (simple resampling)
        if speed != 1.0:
            from scipy import signal

            target_length = int(len(audio_np) / speed)
            audio_np = signal.resample(audio_np, target_length)

        # Calculate duration
        duration = len(audio_np) / config.sample_rate

        return TTSOutput(
            audio=audio_np,
            sample_rate=config.sample_rate,
            duration=duration,
            voice=voice,
            text=text,
        )

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: str = "hi_male",
        speed: float = 1.0,
        normalize_text: bool = True,
    ) -> str:
        """
        Synthesize speech and save to file

        Args:
            text: Input text to synthesize
            output_path: Path to save audio file
            voice: Voice key
            speed: Speech speed multiplier
            normalize_text: Whether to apply text normalization

        Returns:
            Path to saved file
        """
        import soundfile as sf

        output = self.synthesize(text, voice, speed, normalize_text)
        sf.write(output_path, output.audio, output.sample_rate)

        logger.info(f"Saved audio to {output_path} (duration: {output.duration:.2f}s)")
        return output_path

    def get_loaded_voices(self) -> List[str]:
        """Get list of currently loaded voices"""
        return list(self._models.keys())

    def get_available_voices(self) -> Dict[str, Dict]:
        """Get all available voices with their status"""
        voices = {}
        for key, config in LANGUAGE_CONFIGS.items():
            voices[key] = {
                "name": config.name,
                "code": config.code,
                "gender": "male" if "male" in key else "female",
                "loaded": key in self._models,
                "downloaded": self.downloader.get_model_path(key) is not None,
            }
        return voices

    def batch_synthesize(
        self, texts: List[str], voice: str = "hi_male", speed: float = 1.0
    ) -> List[TTSOutput]:
        """Synthesize multiple texts"""
        return [self.synthesize(text, voice, speed) for text in texts]


# Convenience function
def synthesize(
    text: str, voice: str = "hi_male", output_path: Optional[str] = None
) -> Union[TTSOutput, str]:
    """
    Quick synthesis function

    Args:
        text: Text to synthesize
        voice: Voice key
        output_path: If provided, saves to file and returns path

    Returns:
        TTSOutput if no output_path, else path to saved file
    """
    engine = TTSEngine()

    if output_path:
        return engine.synthesize_to_file(text, output_path, voice)
    return engine.synthesize(text, voice)
