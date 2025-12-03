"""
Main TTS Engine for SYSPIN Multi-lingual TTS
Loads and runs VITS models for inference
Includes style/prosody control and Gujarati MMS support
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
import numpy as np
import torch
from dataclasses import dataclass

from .config import LANGUAGE_CONFIGS, LanguageConfig, MODELS_DIR, STYLE_PRESETS
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
    style: Optional[str] = None


class StyleProcessor:
    """
    Simple prosody/style control via audio post-processing
    Supports pitch shifting, speed change, and energy modification
    """

    @staticmethod
    def apply_pitch_shift(
        audio: np.ndarray, sample_rate: int, pitch_factor: float
    ) -> np.ndarray:
        """
        Shift pitch without changing duration using phase vocoder
        pitch_factor > 1.0 = higher pitch, < 1.0 = lower pitch
        """
        if pitch_factor == 1.0:
            return audio

        try:
            import librosa

            # Pitch shift in semitones
            semitones = 12 * np.log2(pitch_factor)
            shifted = librosa.effects.pitch_shift(
                audio.astype(np.float32), sr=sample_rate, n_steps=semitones
            )
            return shifted
        except ImportError:
            # Fallback: simple resampling-based pitch shift (changes duration slightly)
            from scipy import signal

            # Resample to change pitch, then resample back to original length
            stretched = signal.resample(audio, int(len(audio) / pitch_factor))
            return signal.resample(stretched, len(audio))

    @staticmethod
    def apply_speed_change(
        audio: np.ndarray, sample_rate: int, speed_factor: float
    ) -> np.ndarray:
        """
        Change speed/tempo without changing pitch
        speed_factor > 1.0 = faster, < 1.0 = slower
        """
        if speed_factor == 1.0:
            return audio

        try:
            import librosa

            # Time stretch
            stretched = librosa.effects.time_stretch(
                audio.astype(np.float32), rate=speed_factor
            )
            return stretched
        except ImportError:
            # Fallback: simple resampling (will also change pitch)
            from scipy import signal

            target_length = int(len(audio) / speed_factor)
            return signal.resample(audio, target_length)

    @staticmethod
    def apply_energy_change(audio: np.ndarray, energy_factor: float) -> np.ndarray:
        """
        Modify audio energy/volume
        energy_factor > 1.0 = louder, < 1.0 = softer
        """
        if energy_factor == 1.0:
            return audio

        # Apply gain with soft clipping to avoid distortion
        modified = audio * energy_factor

        # Soft clip using tanh for natural sound
        if energy_factor > 1.0:
            max_val = np.max(np.abs(modified))
            if max_val > 0.95:
                modified = np.tanh(modified * 2) * 0.95

        return modified

    @staticmethod
    def apply_style(
        audio: np.ndarray,
        sample_rate: int,
        speed: float = 1.0,
        pitch: float = 1.0,
        energy: float = 1.0,
    ) -> np.ndarray:
        """Apply all style modifications"""
        result = audio

        # Apply in order: pitch -> speed -> energy
        if pitch != 1.0:
            result = StyleProcessor.apply_pitch_shift(result, sample_rate, pitch)

        if speed != 1.0:
            result = StyleProcessor.apply_speed_change(result, sample_rate, speed)

        if energy != 1.0:
            result = StyleProcessor.apply_energy_change(result, energy)

        return result

    @staticmethod
    def get_preset(preset_name: str) -> Dict[str, float]:
        """Get style parameters from preset name"""
        return STYLE_PRESETS.get(preset_name, STYLE_PRESETS["default"])


class TTSEngine:
    """
    Multi-lingual TTS Engine using SYSPIN VITS models

    Supports 11 Indian languages with male/female voices:
    - Hindi, Bengali, Marathi, Telugu, Kannada
    - Bhojpuri, Chhattisgarhi, Maithili, Magahi, English
    - Gujarati (via Facebook MMS)

    Features:
    - Style/prosody control (pitch, speed, energy)
    - Preset styles (happy, sad, calm, excited, etc.)
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

        # MMS models cache (separate handling)
        self._mms_models: Dict[str, any] = {}
        self._mms_tokenizers: Dict[str, any] = {}

        # Downloader
        self.downloader = ModelDownloader(models_dir)

        # Text normalizer
        self.normalizer = TextNormalizer()

        # Style processor
        self.style_processor = StyleProcessor()

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

    def _load_mms_voice(self, voice_key: str) -> bool:
        """
        Load Facebook MMS model for Gujarati
        """
        if voice_key in self._mms_models:
            return True

        config = LANGUAGE_CONFIGS[voice_key]
        logger.info(f"Loading MMS model: {config.hf_model_id}")

        try:
            from transformers import VitsModel, AutoTokenizer

            # Load model and tokenizer from HuggingFace
            model = VitsModel.from_pretrained(config.hf_model_id)
            tokenizer = AutoTokenizer.from_pretrained(config.hf_model_id)

            model = model.to(self.device)
            model.eval()

            self._mms_models[voice_key] = model
            self._mms_tokenizers[voice_key] = tokenizer

            logger.info(f"Loaded MMS voice: {voice_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to load MMS model: {e}")
            raise

    def _synthesize_mms(self, text: str, voice_key: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize using Facebook MMS model (for Gujarati)
        """
        if voice_key not in self._mms_models:
            self._load_mms_voice(voice_key)

        model = self._mms_models[voice_key]
        tokenizer = self._mms_tokenizers[voice_key]
        config = LANGUAGE_CONFIGS[voice_key]

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output = model(**inputs)

        # Get audio
        audio = output.waveform.squeeze().cpu().numpy()
        return audio, config.sample_rate

    def unload_voice(self, voice_key: str):
        """Unload a voice to free memory"""
        if voice_key in self._models:
            del self._models[voice_key]
            del self._tokenizers[voice_key]
        if voice_key in self._mms_models:
            del self._mms_models[voice_key]
            del self._mms_tokenizers[voice_key]
        torch.cuda.empty_cache() if self.device.type == "cuda" else None
        logger.info(f"Unloaded voice: {voice_key}")

    def synthesize(
        self,
        text: str,
        voice: str = "hi_male",
        speed: float = 1.0,
        pitch: float = 1.0,
        energy: float = 1.0,
        style: Optional[str] = None,
        normalize_text: bool = True,
    ) -> TTSOutput:
        """
        Synthesize speech from text with style control

        Args:
            text: Input text to synthesize
            voice: Voice key (e.g., 'hi_male', 'bn_female', 'gu_mms')
            speed: Speech speed multiplier (0.5-2.0)
            pitch: Pitch multiplier (0.5-2.0), >1 = higher
            energy: Energy/volume multiplier (0.5-2.0)
            style: Style preset name (e.g., 'happy', 'sad', 'calm')
            normalize_text: Whether to apply text normalization

        Returns:
            TTSOutput with audio array and metadata
        """
        # Apply style preset if specified
        if style and style in STYLE_PRESETS:
            preset = STYLE_PRESETS[style]
            speed = speed * preset["speed"]
            pitch = pitch * preset["pitch"]
            energy = energy * preset["energy"]

        config = LANGUAGE_CONFIGS[voice]

        # Normalize text
        if normalize_text:
            text = self.normalizer.clean_text(text, config.code)

        # Check if this is an MMS model (Gujarati)
        if "mms" in voice:
            audio_np, sample_rate = self._synthesize_mms(text, voice)
        else:
            # Load voice if not cached (SYSPIN models)
            if voice not in self._models:
                self.load_voice(voice)

            model = self._models[voice]
            tokenizer = self._tokenizers[voice]

            # Tokenize
            token_ids = tokenizer.text_to_ids(text)
            x = torch.from_numpy(np.array(token_ids)).unsqueeze(0).to(self.device)

            # Generate audio
            with torch.no_grad():
                audio = model(x)

            audio_np = audio.squeeze().cpu().numpy()
            sample_rate = config.sample_rate

        # Apply style modifications (pitch, speed, energy)
        audio_np = self.style_processor.apply_style(
            audio_np, sample_rate, speed=speed, pitch=pitch, energy=energy
        )

        # Calculate duration
        duration = len(audio_np) / sample_rate

        return TTSOutput(
            audio=audio_np,
            sample_rate=sample_rate,
            duration=duration,
            voice=voice,
            text=text,
            style=style,
        )

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: str = "hi_male",
        speed: float = 1.0,
        pitch: float = 1.0,
        energy: float = 1.0,
        style: Optional[str] = None,
        normalize_text: bool = True,
    ) -> str:
        """
        Synthesize speech and save to file

        Args:
            text: Input text to synthesize
            output_path: Path to save audio file
            voice: Voice key
            speed: Speech speed multiplier
            pitch: Pitch multiplier
            energy: Energy multiplier
            style: Style preset name
            normalize_text: Whether to apply text normalization

        Returns:
            Path to saved file
        """
        import soundfile as sf

        output = self.synthesize(
            text, voice, speed, pitch, energy, style, normalize_text
        )
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
            is_mms = "mms" in key
            voices[key] = {
                "name": config.name,
                "code": config.code,
                "gender": (
                    "male"
                    if "male" in key
                    else ("female" if "female" in key else "neutral")
                ),
                "loaded": key in self._models or key in self._mms_models,
                "downloaded": is_mms or self.downloader.get_model_path(key) is not None,
                "type": "mms" if is_mms else "vits",
            }
        return voices

    def get_style_presets(self) -> Dict[str, Dict]:
        """Get available style presets"""
        return STYLE_PRESETS

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
