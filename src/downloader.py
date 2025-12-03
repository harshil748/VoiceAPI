"""
Model Downloader for SYSPIN TTS Models
Downloads models from Hugging Face Hub
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

from .config import LANGUAGE_CONFIGS, LanguageConfig, MODELS_DIR

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Downloads and manages SYSPIN TTS models from Hugging Face"""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def download_model(self, voice_key: str, force: bool = False) -> Path:
        """
        Download a specific voice model

        Args:
            voice_key: Key from LANGUAGE_CONFIGS (e.g., 'hi_male', 'bn_female')
            force: Re-download even if exists

        Returns:
            Path to downloaded model directory
        """
        if voice_key not in LANGUAGE_CONFIGS:
            raise ValueError(
                f"Unknown voice: {voice_key}. Available: {list(LANGUAGE_CONFIGS.keys())}"
            )

        config = LANGUAGE_CONFIGS[voice_key]
        model_dir = self.models_dir / voice_key

        # Check if already downloaded
        model_path = model_dir / config.model_filename
        chars_path = model_dir / config.chars_filename
        extra_path = model_dir / "extra.py"

        if not force and model_path.exists() and chars_path.exists():
            logger.info(f"Model {voice_key} already downloaded at {model_dir}")
            return model_dir

        logger.info(f"Downloading {voice_key} from {config.hf_model_id}...")

        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download all files from the repo
            snapshot_download(
                repo_id=config.hf_model_id,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                allow_patterns=["*.pt", "*.pth", "*.txt", "*.py", "*.json"],
            )
            logger.info(f"Successfully downloaded {voice_key} to {model_dir}")

        except Exception as e:
            logger.error(f"Failed to download {voice_key}: {e}")
            raise

        return model_dir

    def download_all_models(self, force: bool = False) -> List[Path]:
        """Download all available models"""
        downloaded = []

        for voice_key in tqdm(LANGUAGE_CONFIGS.keys(), desc="Downloading models"):
            try:
                path = self.download_model(voice_key, force=force)
                downloaded.append(path)
            except Exception as e:
                logger.warning(f"Failed to download {voice_key}: {e}")

        return downloaded

    def download_language(self, lang_code: str, force: bool = False) -> List[Path]:
        """Download all voices for a specific language"""
        downloaded = []

        for voice_key, config in LANGUAGE_CONFIGS.items():
            if config.code == lang_code:
                try:
                    path = self.download_model(voice_key, force=force)
                    downloaded.append(path)
                except Exception as e:
                    logger.warning(f"Failed to download {voice_key}: {e}")

        return downloaded

    def get_model_path(self, voice_key: str) -> Optional[Path]:
        """Get path to a downloaded model"""
        if voice_key not in LANGUAGE_CONFIGS:
            return None

        config = LANGUAGE_CONFIGS[voice_key]
        model_path = self.models_dir / voice_key / config.model_filename

        if model_path.exists():
            return model_path.parent
        return None

    def list_downloaded_models(self) -> List[str]:
        """List all downloaded models"""
        downloaded = []

        for voice_key, config in LANGUAGE_CONFIGS.items():
            model_path = self.models_dir / voice_key / config.model_filename
            if model_path.exists():
                downloaded.append(voice_key)

        return downloaded

    def get_model_size(self, voice_key: str) -> Optional[int]:
        """Get size of downloaded model in bytes"""
        model_path = self.get_model_path(voice_key)
        if not model_path:
            return None

        total_size = 0
        for f in model_path.iterdir():
            if f.is_file():
                total_size += f.stat().st_size

        return total_size


def download_models_cli():
    """CLI entry point for downloading models"""
    import argparse

    parser = argparse.ArgumentParser(description="Download SYSPIN TTS models")
    parser.add_argument(
        "--voice", type=str, help="Specific voice to download (e.g., hi_male)"
    )
    parser.add_argument(
        "--lang", type=str, help="Download all voices for a language (e.g., hi)"
    )
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--force", action="store_true", help="Force re-download")

    args = parser.parse_args()

    downloader = ModelDownloader()

    if args.list:
        print("Available voices:")
        for key, config in LANGUAGE_CONFIGS.items():
            downloaded = "✓" if downloader.get_model_path(key) else " "
            print(f"  [{downloaded}] {key}: {config.name} ({config.code})")
        return

    if args.voice:
        downloader.download_model(args.voice, force=args.force)
    elif args.lang:
        downloader.download_language(args.lang, force=args.force)
    elif args.all:
        downloader.download_all_models(force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    download_models_cli()
