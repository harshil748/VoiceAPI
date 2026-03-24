"""
Main TTS Engine for Multi-lingual Indian Language Speech Synthesis
Loads and runs VITS models for inference entirely from local files.

Supported model types:
  - JIT traced models (.pt + chars.txt)   — Hindi, Bengali, Marathi, Telugu,
                                             Kannada, English, Chhattisgarhi,
                                             Maithili, Magahi
  - Coqui TTS checkpoints (.pth)          — Bhojpuri
  - Facebook MMS (transformers VITS)      — Gujarati  (cached locally)
  - XTTS v2 (Coqui multilingual)          — Voice cloning (cached locally)

All model weights are stored under the project-local `models/` directory.
No runtime HuggingFace downloads are required once models are present.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .config import LANGUAGE_CONFIGS, MODELS_DIR, STYLE_PRESETS
from .tokenizer import TextNormalizer, TTSTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TTSOutput:
    """Output from TTS synthesis"""

    audio: np.ndarray
    sample_rate: int
    duration: float
    voice: str
    text: str
    style: Optional[str] = None


# ---------------------------------------------------------------------------
# Style / Prosody processor
# ---------------------------------------------------------------------------


class StyleProcessor:
    """
    Post-processing based prosody control.
    Applies pitch shift, speed change, and energy modification to raw audio.
    """

    @staticmethod
    def apply_pitch_shift(
        audio: np.ndarray, sample_rate: int, pitch_factor: float
    ) -> np.ndarray:
        """
        Shift pitch without changing duration using librosa phase vocoder.
        pitch_factor > 1.0 → higher pitch, < 1.0 → lower pitch.
        """
        if pitch_factor == 1.0:
            return audio

        try:
            import librosa

            semitones = 12 * np.log2(pitch_factor)
            shifted = librosa.effects.pitch_shift(
                audio.astype(np.float32), sr=sample_rate, n_steps=semitones
            )
            return np.array(shifted, dtype=np.float32)
        except Exception:
            # Catches both ImportError and NumPy compat errors (e.g. librosa 0.10
            # + NumPy 1.x ufunc signature mismatch). Fall back to scipy resampling.
            from scipy import signal

            stretched = signal.resample(audio, int(len(audio) / pitch_factor))
            return np.array(signal.resample(stretched, len(audio)), dtype=np.float32)

    @staticmethod
    def apply_speed_change(
        audio: np.ndarray, sample_rate: int, speed_factor: float
    ) -> np.ndarray:
        """
        Change playback speed without altering pitch.
        speed_factor > 1.0 → faster, < 1.0 → slower.
        """
        if speed_factor == 1.0:
            return audio

        try:
            import librosa

            ts = librosa.effects.time_stretch(
                audio.astype(np.float32), rate=speed_factor
            )
            return np.array(ts, dtype=np.float32)
        except Exception:
            # Catches both ImportError and NumPy compat errors (e.g. librosa 0.10
            # + NumPy 1.x ufunc signature mismatch). Fall back to scipy resampling.
            from scipy import signal

            target_length = int(len(audio) / speed_factor)
            return np.array(signal.resample(audio, target_length), dtype=np.float32)

    @staticmethod
    def apply_energy_change(audio: np.ndarray, energy_factor: float) -> np.ndarray:
        """
        Scale audio amplitude.
        energy_factor > 1.0 → louder, < 1.0 → softer.
        Soft-clips with tanh to avoid distortion on heavy boosts.
        """
        if energy_factor == 1.0:
            return audio

        modified = audio * energy_factor
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
        """Apply pitch → speed → energy in a fixed order."""
        result = audio
        if pitch != 1.0:
            result = StyleProcessor.apply_pitch_shift(result, sample_rate, pitch)
        if speed != 1.0:
            result = StyleProcessor.apply_speed_change(result, sample_rate, speed)
        if energy != 1.0:
            result = StyleProcessor.apply_energy_change(result, energy)
        return result

    @staticmethod
    def get_preset(preset_name: str) -> Dict[str, float]:
        return STYLE_PRESETS.get(preset_name, STYLE_PRESETS["default"])


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class TTSEngine:
    """
    Multi-lingual TTS Engine — 100 % local inference.

    All model files live under `models_dir` (default: project root `models/`).
    On first use of a missing model the engine will attempt a one-time
    download from HuggingFace Hub and cache the files locally so that
    subsequent runs are fully offline.

    Voice cloning uses XTTS v2 (Coqui TTS), whose weights are also stored
    inside `models_dir` (under `tts_models--multilingual--multi-dataset--xtts_v2/`).
    """

    def __init__(
        self,
        models_dir: str = MODELS_DIR,
        device: str = "auto",
        preload_voices: Optional[List[str]] = None,
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._get_device(device)

        # ── Model caches ──────────────────────────────────────────────────
        self._models: Dict[str, torch.jit.ScriptModule] = {}  # JIT .pt
        self._tokenizers: Dict[str, TTSTokenizer] = {}
        self._coqui_models: Dict[str, Any] = {}  # Coqui .pth
        self._mms_models: Dict[str, Any] = {}  # HF VitsModel
        self._mms_tokenizers: Dict[str, Any] = {}
        self._xtts_model: Optional[Any] = None  # XTTS v2

        self.normalizer = TextNormalizer()
        self.style_processor = StyleProcessor()

        if preload_voices:
            for voice in preload_voices:
                try:
                    self.load_voice(voice)
                except Exception as exc:
                    logger.warning("Could not preload voice %s: %s", voice, exc)

        logger.info("TTS Engine ready — device: %s", self.device)

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------

    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            # MPS disabled: TorchScript ops have known compatibility issues
            return torch.device("cpu")
        return torch.device(device)

    # ------------------------------------------------------------------
    # Voice loading (public)
    # ------------------------------------------------------------------

    def load_voice(self, voice_key: str, download_if_missing: bool = True) -> bool:
        """
        Load a voice model into memory.

        Args:
            voice_key: Key from LANGUAGE_CONFIGS (e.g. 'hi_male', 'gu_mms').
            download_if_missing: Attempt HuggingFace download when files absent.

        Returns:
            True on success.
        """
        if voice_key in self._models or voice_key in self._coqui_models:
            return True

        if voice_key not in LANGUAGE_CONFIGS:
            raise ValueError(f"Unknown voice key: '{voice_key}'")

        # MMS models have their own loading path
        if "mms" in voice_key:
            return self._load_mms_voice(voice_key, download_if_missing)

        model_dir = self.models_dir / voice_key

        # Download if the directory doesn't exist yet
        if not model_dir.exists() or not any(model_dir.iterdir()):
            if download_if_missing:
                logger.info("Model files missing for %s — downloading…", voice_key)
                self._download_syspin_model(voice_key)
            else:
                raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Decide between Coqui checkpoint (.pth) and JIT trace (.pt)
        pth_files = list(model_dir.glob("*.pth"))
        pt_files = list(model_dir.glob("*.pt"))

        if pth_files:
            return self._load_coqui_voice(voice_key, model_dir, pth_files[0])
        elif pt_files:
            return self._load_jit_voice(voice_key, model_dir, pt_files[0])
        else:
            raise FileNotFoundError(f"No .pt or .pth model file found in {model_dir}")

    # ------------------------------------------------------------------
    # JIT traced models (.pt)
    # ------------------------------------------------------------------

    def _load_jit_voice(
        self, voice_key: str, model_dir: Path, model_path: Path
    ) -> bool:
        # Locate chars.txt
        chars_path = model_dir / "chars.txt"
        if not chars_path.exists():
            candidates = list(model_dir.glob("*chars*.txt"))
            if not candidates:
                raise FileNotFoundError(f"No chars.txt found in {model_dir}")
            chars_path = candidates[0]

        tokenizer = TTSTokenizer.from_chars_file(str(chars_path))

        logger.info("Loading JIT model: %s", model_path)
        model = torch.jit.load(str(model_path), map_location=self.device)
        model.eval()

        self._models[voice_key] = model
        self._tokenizers[voice_key] = tokenizer
        logger.info("Loaded JIT voice: %s", voice_key)
        return True

    # ------------------------------------------------------------------
    # Coqui TTS checkpoints (.pth)
    # ------------------------------------------------------------------

    def _load_coqui_voice(
        self, voice_key: str, model_dir: Path, checkpoint_path: Path
    ) -> bool:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json found in {model_dir}")

        try:
            from TTS.utils.synthesizer import Synthesizer
        except ImportError:
            raise ImportError(
                "Coqui TTS library is required for Bhojpuri models.\n"
                "Install it with:  pip install TTS"
            )

        logger.info("Loading Coqui checkpoint: %s", checkpoint_path)
        use_cuda = self.device.type == "cuda"
        synthesizer = Synthesizer(
            tts_checkpoint=str(checkpoint_path),
            tts_config_path=str(config_path),
            use_cuda=use_cuda,
        )
        self._coqui_models[voice_key] = synthesizer
        logger.info("Loaded Coqui voice: %s", voice_key)
        return True

    def _synthesize_coqui(self, text: str, voice_key: str) -> Tuple[np.ndarray, int]:
        if voice_key not in self._coqui_models:
            self.load_voice(voice_key)
        synthesizer = self._coqui_models[voice_key]
        wav = synthesizer.tts(text)
        return np.array(wav, dtype=np.float32), synthesizer.output_sample_rate

    # ------------------------------------------------------------------
    # Facebook MMS (Gujarati, transformers VitsModel)
    # ------------------------------------------------------------------

    def _load_mms_voice(self, voice_key: str, download_if_missing: bool = True) -> bool:
        """
        Load a Facebook MMS model.

        Looks for model weights in `models/<voice_key>/` first.
        If the weights file (model.safetensors or pytorch_model.bin) is absent
        and `download_if_missing` is True, downloads the full snapshot from
        HuggingFace Hub into that directory so future runs are fully offline.
        """
        if voice_key in self._mms_models:
            return True

        config = LANGUAGE_CONFIGS[voice_key]
        local_dir = self.models_dir / voice_key
        local_dir.mkdir(parents=True, exist_ok=True)

        _WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
        has_weights = any((local_dir / f).exists() for f in _WEIGHT_FILES)

        if not has_weights:
            if not download_if_missing:
                raise FileNotFoundError(
                    f"MMS model weights not found in {local_dir}. "
                    "Run with download_if_missing=True to fetch them."
                )
            logger.info(
                "MMS weights missing for %s — downloading from %s …",
                voice_key,
                config.hf_model_id,
            )
            self._download_hf_snapshot(config.hf_model_id, local_dir)

        try:
            from transformers import AutoTokenizer, VitsModel
        except ImportError:
            raise ImportError(
                "The `transformers` library is required for Gujarati (MMS).\n"
                "Install it with:  pip install transformers"
            )

        logger.info("Loading MMS model from local: %s", local_dir)
        model = VitsModel.from_pretrained(str(local_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(local_dir))

        model = model.to(self.device)  # type: ignore[arg-type]
        model.eval()

        self._mms_models[voice_key] = model
        self._mms_tokenizers[voice_key] = tokenizer
        logger.info("Loaded MMS voice: %s", voice_key)
        return True

    def _synthesize_mms(self, text: str, voice_key: str) -> Tuple[np.ndarray, int]:
        if voice_key not in self._mms_models:
            self._load_mms_voice(voice_key)

        model = self._mms_models[voice_key]
        tokenizer = self._mms_tokenizers[voice_key]
        config = LANGUAGE_CONFIGS[voice_key]

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        audio = output.waveform.squeeze().cpu().numpy()
        return audio, config.sample_rate

    # ------------------------------------------------------------------
    # XTTS v2 — voice cloning
    # ------------------------------------------------------------------

    def _get_xtts_model(self):
        """
        Lazy-load Coqui XTTS v2.

        The model weights (~1.8 GB) are cached inside `models_dir` by
        setting the COQUI_TTS_HOME environment variable before the first
        call to TTS().  Subsequent runs load entirely from disk.

        Cache location: <models_dir>/tts_models--multilingual--multi-dataset--xtts_v2/
        """
        if self._xtts_model is not None:
            return self._xtts_model

        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise ImportError(
                "Coqui TTS library is required for voice cloning.\n"
                "Install it with:  pip install TTS"
            ) from exc

        # Point Coqui TTS cache into our local models directory
        os.environ["COQUI_TTS_HOME"] = str(self.models_dir)

        logger.info(
            "Loading XTTS v2 voice cloning model "
            "(will download ~1.8 GB on first use, cached at %s) …",
            self.models_dir,
        )
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        if self.device.type == "cuda":
            tts = tts.to("cuda")

        self._xtts_model = tts
        logger.info("XTTS v2 loaded successfully")
        return self._xtts_model

    def clone_voice(
        self,
        text: str,
        speaker_wav_path: str,
        language_code: str = "en",
        speed: float = 1.0,
        pitch: float = 1.0,
        energy: float = 1.0,
        style: Optional[str] = None,
        normalize_text: bool = True,
    ) -> TTSOutput:
        """
        Clone a speaker's voice from a reference WAV file using XTTS v2.

        Args:
            text:              Text to synthesize in the cloned voice.
            speaker_wav_path:  Path to reference WAV (3–30 s recommended).
            language_code:     XTTS language code (en, hi, bn, mr, te, kn, gu).
            speed/pitch/energy: Post-processing prosody controls (0.5–2.0).
            style:             Optional style-preset name.
            normalize_text:    Apply text normalisation before synthesis.

        Returns:
            TTSOutput with 24 kHz audio (XTTS default).
        """
        xtts = self._get_xtts_model()

        if normalize_text:
            text = self.normalizer.clean_text(text, language_code)

        # XTTS inference
        wav = xtts.tts(
            text=text,
            speaker_wav=speaker_wav_path,
            language=language_code,
        )
        audio_np = np.array(wav, dtype=np.float32)
        sample_rate = 24_000  # XTTS always outputs 24 kHz

        # Apply style-preset multipliers on top of explicit params
        if style and style in STYLE_PRESETS:
            preset = STYLE_PRESETS[style]
            speed *= preset["speed"]
            pitch *= preset["pitch"]
            energy *= preset["energy"]

        audio_np = self.style_processor.apply_style(
            audio_np, sample_rate, speed=speed, pitch=pitch, energy=energy
        )

        duration = len(audio_np) / sample_rate
        return TTSOutput(
            audio=audio_np,
            sample_rate=sample_rate,
            duration=duration,
            voice="custom_cloned",
            text=text,
            style=style,
        )

    # ------------------------------------------------------------------
    # Standard synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice: str = "hi_female",
        speed: float = 1.0,
        pitch: float = 1.0,
        energy: float = 1.0,
        style: Optional[str] = None,
        normalize_text: bool = True,
    ) -> TTSOutput:
        """
        Synthesize speech from text.

        Args:
            text:          Input text.
            voice:         Voice key (e.g. 'hi_male', 'bn_female', 'gu_mms').
            speed:         Speed multiplier (0.5–2.0).
            pitch:         Pitch multiplier (0.5–2.0).
            energy:        Amplitude multiplier (0.5–2.0).
            style:         Style preset name.
            normalize_text: Apply text normalisation.

        Returns:
            TTSOutput containing audio array and metadata.
        """
        if voice not in LANGUAGE_CONFIGS:
            raise ValueError(
                f"Unknown voice: '{voice}'. "
                f"Available: {sorted(LANGUAGE_CONFIGS.keys())}"
            )

        config = LANGUAGE_CONFIGS[voice]

        # Resolve style preset
        if style and style in STYLE_PRESETS:
            preset = STYLE_PRESETS[style]
            speed *= preset["speed"]
            pitch *= preset["pitch"]
            energy *= preset["energy"]

        # Text normalisation
        if normalize_text:
            text = self.normalizer.clean_text(text, config.code)

        # ── Inference ────────────────────────────────────────────────
        if "mms" in voice:
            audio_np, sample_rate = self._synthesize_mms(text, voice)

        elif voice in self._coqui_models:
            audio_np, sample_rate = self._synthesize_coqui(text, voice)

        else:
            # Load model if not in cache yet (handles both JIT and Coqui)
            if voice not in self._models and voice not in self._coqui_models:
                self.load_voice(voice)

            if voice in self._coqui_models:
                audio_np, sample_rate = self._synthesize_coqui(text, voice)
            else:
                model = self._models[voice]
                tokenizer = self._tokenizers[voice]

                token_ids = tokenizer.text_to_ids(text)
                x = torch.from_numpy(np.array(token_ids)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    audio = model(x)

                audio_np = audio.squeeze().cpu().numpy()
                sample_rate = config.sample_rate

        # ── Style post-processing ────────────────────────────────────
        audio_np = self.style_processor.apply_style(
            audio_np, sample_rate, speed=speed, pitch=pitch, energy=energy
        )

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
        voice: str = "hi_female",
        speed: float = 1.0,
        pitch: float = 1.0,
        energy: float = 1.0,
        style: Optional[str] = None,
        normalize_text: bool = True,
    ) -> str:
        """Synthesize and save to a WAV file. Returns the saved path."""
        import soundfile as sf

        output = self.synthesize(
            text, voice, speed, pitch, energy, style, normalize_text
        )
        sf.write(output_path, output.audio, output.sample_rate)
        logger.info("Saved audio → %s (%.2f s)", output_path, output.duration)
        return output_path

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def unload_voice(self, voice_key: str) -> None:
        """Remove a voice model from memory."""
        removed = False
        if voice_key in self._models:
            del self._models[voice_key]
            del self._tokenizers[voice_key]
            removed = True
        if voice_key in self._coqui_models:
            del self._coqui_models[voice_key]
            removed = True
        if voice_key in self._mms_models:
            del self._mms_models[voice_key]
            del self._mms_tokenizers[voice_key]
            removed = True
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if removed:
            logger.info("Unloaded voice: %s", voice_key)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_loaded_voices(self) -> List[str]:
        """Return list of currently loaded voice keys."""
        return (
            list(self._models.keys())
            + list(self._coqui_models.keys())
            + list(self._mms_models.keys())
        )

    def get_available_voices(self) -> Dict[str, Dict]:
        """Return all configured voices with download / load status."""
        voices: Dict[str, Dict] = {}
        for key, config in LANGUAGE_CONFIGS.items():
            is_mms = "mms" in key
            model_dir = self.models_dir / key

            if is_mms:
                model_type = "mms"
                downloaded = any(
                    (model_dir / f).exists()
                    for f in ("model.safetensors", "pytorch_model.bin")
                )
            elif model_dir.exists() and list(model_dir.glob("*.pth")):
                model_type = "coqui"
                downloaded = True
            elif model_dir.exists() and list(model_dir.glob("*.pt")):
                model_type = "vits_jit"
                downloaded = True
            else:
                model_type = "vits_jit"
                downloaded = False

            voices[key] = {
                "name": config.name,
                "code": config.code,
                "gender": (
                    "male"
                    if "male" in key
                    else "female"
                    if "female" in key
                    else "neutral"
                ),
                "loaded": (
                    key in self._models
                    or key in self._coqui_models
                    or key in self._mms_models
                ),
                "downloaded": downloaded,
                "type": model_type,
            }
        return voices

    def get_style_presets(self) -> Dict[str, Dict]:
        return STYLE_PRESETS

    def batch_synthesize(
        self,
        texts: List[str],
        voice: str = "hi_female",
        speed: float = 1.0,
    ) -> List[TTSOutput]:
        """Synthesize a list of texts sequentially."""
        return [self.synthesize(text, voice, speed) for text in texts]

    # ------------------------------------------------------------------
    # Bulk preload
    # ------------------------------------------------------------------

    def preload_all_voices(self) -> Dict[str, bool]:
        """
        Download (if missing) and load every configured synthesis voice into memory.

        XTTS v2 (voice cloning) is intentionally excluded — it is lazy-loaded
        on the first /clone request so startup stays fast.

        Returns:
            Dict mapping voice_key → True (loaded OK) / False (failed).
        """
        results: Dict[str, bool] = {}

        for voice_key, config in LANGUAGE_CONFIGS.items():
            try:
                # Ensure model files are present on disk before loading
                if "mms" in voice_key:
                    local_dir = self.models_dir / voice_key
                    _WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
                    if not any((local_dir / f).exists() for f in _WEIGHT_FILES):
                        logger.info(
                            "Preload: MMS weights missing for %s — downloading…",
                            voice_key,
                        )
                        self._download_hf_snapshot(config.hf_model_id, local_dir)
                else:
                    model_dir = self.models_dir / voice_key
                    if not model_dir.exists() or not any(model_dir.iterdir()):
                        logger.info(
                            "Preload: model files missing for %s — downloading…",
                            voice_key,
                        )
                        self._download_syspin_model(voice_key)

                # Load the voice into RAM (no-op if already loaded)
                self.load_voice(voice_key, download_if_missing=True)
                results[voice_key] = True
                logger.info("Preload OK: %s", voice_key)

            except Exception as exc:
                logger.warning("Preload failed for voice '%s': %s", voice_key, exc)
                results[voice_key] = False

        return results

    # ------------------------------------------------------------------
    # Internal download helpers
    # ------------------------------------------------------------------

    def _download_syspin_model(self, voice_key: str) -> None:
        """Download a SYSPIN model snapshot from HuggingFace Hub."""
        from huggingface_hub import snapshot_download

        config = LANGUAGE_CONFIGS[voice_key]
        model_dir = self.models_dir / voice_key
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading %s from HuggingFace: %s …", voice_key, config.hf_model_id
        )
        snapshot_download(
            repo_id=config.hf_model_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["*.pt", "*.pth", "*.txt", "*.py", "*.json"],
        )
        logger.info("Download complete: %s", model_dir)

    def _download_hf_snapshot(self, repo_id: str, local_dir: Path) -> None:
        """Generic HuggingFace snapshot download into a local directory."""
        from huggingface_hub import snapshot_download

        local_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s → %s …", repo_id, local_dir)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        logger.info("Download complete: %s", local_dir)


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------


def synthesize(
    text: str,
    voice: str = "hi_female",
    output_path: Optional[str] = None,
) -> Union[TTSOutput, str]:
    """
    One-shot synthesis helper.

    Args:
        text:        Text to synthesize.
        voice:       Voice key.
        output_path: If given, saves WAV to this path and returns the path.

    Returns:
        TTSOutput (in-memory) or str (file path).
    """
    engine = TTSEngine()
    if output_path:
        return engine.synthesize_to_file(text, output_path, voice)
    return engine.synthesize(text, voice)
