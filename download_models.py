#!/usr/bin/env python3
"""
Download all required TTS models from HuggingFace
Run this on deployment to fetch models before starting the server
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.downloader import ModelDownloader
from src.config import LANGUAGE_CONFIGS


def main():
    print("=" * 60)
    print("Downloading TTS Models from HuggingFace...")
    print("=" * 60)

    downloader = ModelDownloader()

    # Download all configured models
    voices = list(LANGUAGE_CONFIGS.keys())
    print(f"\nModels to download: {len(voices)}")
    for v in voices:
        print(f"  - {v}")

    print("\n")

    success = 0
    failed = []

    for voice in voices:
        try:
            print(f"Downloading {voice}...")
            downloader.download_model(voice)
            success += 1
            print(f"  ✓ {voice} downloaded\n")
        except Exception as e:
            print(f"  ✗ {voice} failed: {e}\n")
            failed.append(voice)

    print("=" * 60)
    print(f"Download complete: {success}/{len(voices)} models")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return 1
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
