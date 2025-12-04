#!/usr/bin/env python3
"""
Dataset Preparation Script for Indian Language TTS Training

This script prepares speech datasets for training VITS models on Indian languages.
It handles data from multiple sources and creates a unified format.

Supported Datasets:
- OpenSLR Indian Language Datasets
- Mozilla Common Voice (Indian subsets)
- IndicTTS Dataset (IIT Madras)
- Custom recordings

Output Format:
- audio/: Normalized WAV files (22050Hz, mono, 16-bit)
- metadata.csv: text|audio_path|speaker_id|duration
"""

import os
import sys
import csv
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Try to import audio processing libraries
try:
    import librosa
    import soundfile as sf

    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("Warning: librosa/soundfile not installed. Audio processing disabled.")


# Dataset configurations
DATASET_CONFIGS = {
    "openslr_hindi": {
        "url": "https://www.openslr.org/resources/103/",
        "name": "OpenSLR Hindi ASR Corpus",
        "language": "hindi",
        "sample_rate": 16000,
    },
    "openslr_bengali": {
        "url": "https://www.openslr.org/resources/37/",
        "name": "OpenSLR Bengali Multi-speaker",
        "language": "bengali",
        "sample_rate": 16000,
    },
    "openslr_marathi": {
        "url": "https://www.openslr.org/resources/64/",
        "name": "OpenSLR Marathi",
        "language": "marathi",
        "sample_rate": 16000,
    },
    "openslr_telugu": {
        "url": "https://www.openslr.org/resources/66/",
        "name": "OpenSLR Telugu",
        "language": "telugu",
        "sample_rate": 16000,
    },
    "openslr_kannada": {
        "url": "https://www.openslr.org/resources/79/",
        "name": "OpenSLR Kannada",
        "language": "kannada",
        "sample_rate": 16000,
    },
    "openslr_gujarati": {
        "url": "https://www.openslr.org/resources/78/",
        "name": "OpenSLR Gujarati",
        "language": "gujarati",
        "sample_rate": 16000,
    },
    "commonvoice_hindi": {
        "url": "https://commonvoice.mozilla.org/en/datasets",
        "name": "Mozilla Common Voice Hindi",
        "language": "hindi",
        "sample_rate": 48000,
    },
    "indictts": {
        "url": "https://www.iitm.ac.in/donlab/tts/",
        "name": "IndicTTS Dataset (IIT Madras)",
        "languages": ["hindi", "bengali", "marathi", "telugu", "kannada", "gujarati"],
        "sample_rate": 22050,
    },
}


@dataclass
class AudioSample:
    """Represents a single audio sample"""

    audio_path: Path
    text: str
    speaker_id: str
    language: str
    duration: float = 0.0
    sample_rate: int = 22050


class DatasetProcessor:
    """Process and prepare datasets for TTS training"""

    TARGET_SAMPLE_RATE = 22050
    MIN_DURATION = 0.5  # seconds
    MAX_DURATION = 15.0  # seconds

    def __init__(self, output_dir: Path, language: str):
        self.output_dir = output_dir
        self.language = language
        self.audio_dir = output_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_audio(self, input_path: Path, output_path: Path) -> Optional[float]:
        """
        Process a single audio file:
        - Resample to target sample rate
        - Convert to mono
        - Normalize volume
        - Trim silence
        """
        if not HAS_AUDIO:
            return None

        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=None, mono=True)

            # Resample if necessary
            if sr != self.TARGET_SAMPLE_RATE:
                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=self.TARGET_SAMPLE_RATE
                )

            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)

            # Normalize
            audio = audio / np.abs(audio).max() * 0.95

            # Calculate duration
            duration = len(audio) / self.TARGET_SAMPLE_RATE

            # Filter by duration
            if duration < self.MIN_DURATION or duration > self.MAX_DURATION:
                return None

            # Save processed audio
            sf.write(output_path, audio, self.TARGET_SAMPLE_RATE)

            return duration

        except Exception as e:
            self.logger.warning(f"Error processing {input_path}: {e}")
            return None

    def process_openslr(self, data_dir: Path) -> List[AudioSample]:
        """Process OpenSLR format dataset"""
        samples = []

        # OpenSLR typically has transcripts.txt or similar
        transcript_file = data_dir / "transcripts.txt"
        if not transcript_file.exists():
            transcript_file = data_dir / "text"

        if transcript_file.exists():
            with open(transcript_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        audio_id, text = parts[0], parts[1]
                        audio_path = data_dir / "audio" / f"{audio_id}.wav"

                        if audio_path.exists():
                            output_path = self.audio_dir / f"{audio_id}.wav"
                            duration = self.process_audio(audio_path, output_path)

                            if duration:
                                samples.append(
                                    AudioSample(
                                        audio_path=output_path,
                                        text=text,
                                        speaker_id="spk_001",
                                        language=self.language,
                                        duration=duration,
                                    )
                                )

        return samples

    def process_commonvoice(self, data_dir: Path) -> List[AudioSample]:
        """Process Mozilla Common Voice format"""
        samples = []

        # Common Voice uses validated.tsv
        tsv_file = data_dir / "validated.tsv"
        clips_dir = data_dir / "clips"

        if tsv_file.exists():
            with open(tsv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    audio_path = clips_dir / row["path"]
                    text = row["sentence"]
                    speaker_id = row.get("client_id", "unknown")[:8]

                    if audio_path.exists():
                        output_name = f"cv_{audio_path.stem}.wav"
                        output_path = self.audio_dir / output_name
                        duration = self.process_audio(audio_path, output_path)

                        if duration:
                            samples.append(
                                AudioSample(
                                    audio_path=output_path,
                                    text=text,
                                    speaker_id=speaker_id,
                                    language=self.language,
                                    duration=duration,
                                )
                            )

        return samples

    def process_indictts(self, data_dir: Path) -> List[AudioSample]:
        """Process IndicTTS format dataset"""
        samples = []

        # IndicTTS has wav/ folder and txt/ folder
        wav_dir = data_dir / "wav"
        txt_dir = data_dir / "txt"

        if wav_dir.exists() and txt_dir.exists():
            for wav_file in wav_dir.glob("*.wav"):
                txt_file = txt_dir / f"{wav_file.stem}.txt"

                if txt_file.exists():
                    with open(txt_file, "r", encoding="utf-8") as f:
                        text = f.read().strip()

                    output_path = self.audio_dir / wav_file.name
                    duration = self.process_audio(wav_file, output_path)

                    if duration:
                        samples.append(
                            AudioSample(
                                audio_path=output_path,
                                text=text,
                                speaker_id="indic_001",
                                language=self.language,
                                duration=duration,
                            )
                        )

        return samples

    def save_metadata(self, samples: List[AudioSample]):
        """Save processed samples to metadata CSV"""
        metadata_path = self.output_dir / "metadata.csv"

        with open(metadata_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["audio_path", "text", "speaker_id", "duration"])

            for sample in samples:
                writer.writerow(
                    [
                        sample.audio_path.name,
                        sample.text,
                        sample.speaker_id,
                        f"{sample.duration:.3f}",
                    ]
                )

        self.logger.info(f"Saved {len(samples)} samples to {metadata_path}")

        # Save statistics
        stats = {
            "total_samples": len(samples),
            "total_duration_hours": sum(s.duration for s in samples) / 3600,
            "language": self.language,
            "speakers": len(set(s.speaker_id for s in samples)),
        }

        with open(self.output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Dataset stats: {stats}")


def create_train_val_split(metadata_path: Path, train_ratio: float = 0.95):
    """Split metadata into train and validation sets"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)
        rows = list(reader)

    # Shuffle
    np.random.shuffle(rows)

    # Split
    split_idx = int(len(rows) * train_ratio)
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    # Save splits
    for name, data in [("train", train_rows), ("val", val_rows)]:
        output_path = metadata_path.parent / f"metadata_{name}.csv"
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(header)
            writer.writerows(data)

        print(f"Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for TTS training")
    parser.add_argument(
        "--input", type=str, required=True, help="Input dataset directory"
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--language", type=str, required=True, help="Target language")
    parser.add_argument(
        "--format",
        type=str,
        default="openslr",
        choices=["openslr", "commonvoice", "indictts"],
        help="Dataset format",
    )
    parser.add_argument("--split", action="store_true", help="Create train/val split")

    args = parser.parse_args()

    processor = DatasetProcessor(
        output_dir=Path(args.output),
        language=args.language,
    )

    # Process based on format
    if args.format == "openslr":
        samples = processor.process_openslr(Path(args.input))
    elif args.format == "commonvoice":
        samples = processor.process_commonvoice(Path(args.input))
    elif args.format == "indictts":
        samples = processor.process_indictts(Path(args.input))

    # Save metadata
    processor.save_metadata(samples)

    # Create train/val split if requested
    if args.split:
        create_train_val_split(Path(args.output) / "metadata.csv")


if __name__ == "__main__":
    main()
