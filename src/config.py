"""
Configuration for SYSPIN Multi-lingual TTS System
"""

from dataclasses import dataclass
from typing import Dict, Optional
import os

# Base path for models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


@dataclass
class LanguageConfig:
    """Configuration for each language"""

    name: str
    code: str
    hf_model_id: str
    model_filename: str
    chars_filename: str = "chars.txt"
    sample_rate: int = 22050


# All SYSPIN models available
# JIT traced format (.pt + chars.txt): Hindi, Bengali, Marathi, Telugu, Kannada, etc.
# Coqui TTS checkpoints (.pth + config.json): Bhojpuri
LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {
    # Hindi
    "hi_male": LanguageConfig(
        name="Hindi",
        code="hi",
        hf_model_id="SYSPIN/tts_vits_coquiai_HindiMale",
        model_filename="hi_male_vits_30hrs.pt",
    ),
    "hi_female": LanguageConfig(
        name="Hindi",
        code="hi",
        hf_model_id="SYSPIN/tts_vits_coquiai_HindiFemale",
        model_filename="hi_female_vits_30hrs.pt",
    ),
    # Bengali
    "bn_male": LanguageConfig(
        name="Bengali",
        code="bn",
        hf_model_id="SYSPIN/tts_vits_coquiai_BengaliMale",
        model_filename="bn_male_vits_30hrs.pt",
    ),
    "bn_female": LanguageConfig(
        name="Bengali",
        code="bn",
        hf_model_id="SYSPIN/tts_vits_coquiai_BengaliFemale",
        model_filename="bn_female_vits_30hrs.pt",
    ),
    # Marathi
    "mr_male": LanguageConfig(
        name="Marathi",
        code="mr",
        hf_model_id="SYSPIN/tts_vits_coquiai_MarathiMale",
        model_filename="mr_male_vits_30hrs.pt",
    ),
    "mr_female": LanguageConfig(
        name="Marathi",
        code="mr",
        hf_model_id="SYSPIN/tts_vits_coquiai_MarathiFemale",
        model_filename="mr_female_vits_30hrs.pt",
    ),
    # Telugu
    "te_male": LanguageConfig(
        name="Telugu",
        code="te",
        hf_model_id="SYSPIN/tts_vits_coquiai_TeluguMale",
        model_filename="te_male_vits_30hrs.pt",
    ),
    "te_female": LanguageConfig(
        name="Telugu",
        code="te",
        hf_model_id="SYSPIN/tts_vits_coquiai_TeluguFemale",
        model_filename="te_female_vits_30hrs.pt",
    ),
    # Kannada
    "kn_male": LanguageConfig(
        name="Kannada",
        code="kn",
        hf_model_id="SYSPIN/tts_vits_coquiai_KannadaMale",
        model_filename="kn_male_vits_30hrs.pt",
    ),
    "kn_female": LanguageConfig(
        name="Kannada",
        code="kn",
        hf_model_id="SYSPIN/tts_vits_coquiai_KannadaFemale",
        model_filename="kn_female_vits_30hrs.pt",
    ),
    # Bhojpuri (Coqui TTS checkpoint format)
    "bho_male": LanguageConfig(
        name="Bhojpuri",
        code="bho",
        hf_model_id="SYSPIN/tts_vits_coquiai_BhojpuriMale",
        model_filename="checkpoint_200000.pth",
    ),
    "bho_female": LanguageConfig(
        name="Bhojpuri",
        code="bho",
        hf_model_id="SYSPIN/tts_vits_coquiai_BhojpuriFemale",
        model_filename="checkpoint_340000.pth",
    ),
    # Chhattisgarhi (ISO 639-3: hne)
    "hne_male": LanguageConfig(
        name="Chhattisgarhi",
        code="hne",
        hf_model_id="SYSPIN/tts_vits_coquiai_ChhattisgarhiMale",
        model_filename="ch_male_vits_30hrs.pt",
    ),
    "hne_female": LanguageConfig(
        name="Chhattisgarhi",
        code="hne",
        hf_model_id="SYSPIN/tts_vits_coquiai_ChhattisgarhiFemale",
        model_filename="ch_female_vits_30hrs.pt",
    ),
    # Maithili (ISO 639-3: mai)
    "mai_male": LanguageConfig(
        name="Maithili",
        code="mai",
        hf_model_id="SYSPIN/tts_vits_coquiai_MaithiliMale",
        model_filename="mt_male_vits_30hrs.pt",
    ),
    "mai_female": LanguageConfig(
        name="Maithili",
        code="mai",
        hf_model_id="SYSPIN/tts_vits_coquiai_MaithiliFemale",
        model_filename="mt_female_vits_30hrs.pt",
    ),
    # Magahi (ISO 639-3: mag)
    "mag_male": LanguageConfig(
        name="Magahi",
        code="mag",
        hf_model_id="SYSPIN/tts_vits_coquiai_MagahiMale",
        model_filename="mg_male_vits_30hrs.pt",
    ),
    "mag_female": LanguageConfig(
        name="Magahi",
        code="mag",
        hf_model_id="SYSPIN/tts_vits_coquiai_MagahiFemale",
        model_filename="mg_female_vits_30hrs.pt",
    ),
    # English
    "en_male": LanguageConfig(
        name="English",
        code="en",
        hf_model_id="SYSPIN/tts_vits_coquiai_EnglishMale",
        model_filename="en_male_vits_30hrs.pt",
    ),
    "en_female": LanguageConfig(
        name="English",
        code="en",
        hf_model_id="SYSPIN/tts_vits_coquiai_EnglishFemale",
        model_filename="en_female_vits_30hrs.pt",
    ),
    # Gujarati - Using Facebook MMS model (1100+ languages)
    "gu_mms": LanguageConfig(
        name="Gujarati",
        code="gu",
        hf_model_id="facebook/mms-tts-guj",
        model_filename="mms_guj.pt",
        sample_rate=16000,  # MMS uses 16kHz
    ),
}


# Style presets for prosody control
STYLE_PRESETS = {
    "default": {"speed": 1.0, "pitch": 1.0, "energy": 1.0},
    "slow": {"speed": 0.75, "pitch": 1.0, "energy": 1.0},
    "fast": {"speed": 1.25, "pitch": 1.0, "energy": 1.0},
    "soft": {"speed": 0.9, "pitch": 0.95, "energy": 0.7},
    "loud": {"speed": 1.0, "pitch": 1.05, "energy": 1.3},
    "happy": {"speed": 1.1, "pitch": 1.1, "energy": 1.2},
    "sad": {"speed": 0.85, "pitch": 0.9, "energy": 0.8},
    "calm": {"speed": 0.9, "pitch": 0.95, "energy": 0.85},
    "excited": {"speed": 1.2, "pitch": 1.15, "energy": 1.3},
}


def get_available_languages() -> Dict[str, str]:
    """Returns mapping of language codes to names"""
    seen = {}
    for key, config in LANGUAGE_CONFIGS.items():
        if config.code not in seen:
            seen[config.code] = config.name
    return seen


def get_available_voices() -> Dict[str, Dict]:
    """Returns all available voice configurations"""
    return {
        key: {
            "name": config.name,
            "code": config.code,
            "gender": (
                "male"
                if "male" in key
                else ("female" if "female" in key else "neutral")
            ),
        }
        for key, config in LANGUAGE_CONFIGS.items()
    }


def get_style_presets() -> Dict[str, Dict]:
    """Returns available style presets"""
    return STYLE_PRESETS
