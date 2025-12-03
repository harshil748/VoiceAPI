# Voice Tech for All - Multi-lingual TTS System

A lightweight, multi-lingual Text-to-Speech system supporting 10+ Indian languages with REST API.

## 🎯 Hackathon: Voice Tech for All

Built for the healthcare assistant use case - helping pregnant mothers in low-income communities access healthcare information in their native languages.

## ✨ Features

- **10+ Indian Languages**: Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Chhattisgarhi, Maithili, Magahi, English
- **Male & Female Voices**: 20 voice options
- **Lightweight**: VITS-based models optimized for fast inference
- **REST API**: FastAPI-powered server with OpenAPI docs
- **Text Normalization**: Handles numbers, punctuation for Indian scripts
- **Apple Silicon Support**: Runs on M1/M2/M3 Macs with MPS acceleration

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and navigate
cd /path/to/TTS

# Create virtual environment (if not exists)
python3 -m venv tts
source tts/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Download Hindi models (male + female)
python -m src.cli download --lang hi

# Or download a specific voice
python -m src.cli download --voice hi_male

# Or download ALL models (~6GB)
python -m src.cli download --all
```

### 3. Synthesize Speech

```bash
# Command line
python -m src.cli synthesize --text "नमस्ते, मैं आपकी मदद कर सकता हूं" --voice hi_male --output hello.wav

# Play the audio (macOS)
afplay hello.wav
```

### 4. Start API Server

```bash
python -m src.cli serve --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## 📡 API Usage

### Synthesize Speech (POST)

```bash
curl -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते", "voice": "hi_female", "speed": 1.0}' \
  --output speech.wav
```

### Synthesize Speech (GET - for testing)

```bash
curl "http://localhost:8000/synthesize/get?text=नमस्ते&voice=hi_male" --output speech.wav
```

### List Available Voices

```bash
curl http://localhost:8000/voices
```

## 🎤 Available Voices

| Language      | Code | Male        | Female        |
| ------------- | ---- | ----------- | ------------- |
| Hindi         | hi   | ✅ hi_male  | ✅ hi_female  |
| Bengali       | bn   | ✅ bn_male  | ✅ bn_female  |
| Marathi       | mr   | ✅ mr_male  | ✅ mr_female  |
| Telugu        | te   | ✅ te_male  | ✅ te_female  |
| Kannada       | kn   | ✅ kn_male  | ✅ kn_female  |
| Bhojpuri      | bho  | ✅ bho_male | ✅ bho_female |
| Chhattisgarhi | hne  | ✅ hne_male | ✅ hne_female |
| Maithili      | mai  | ✅ mai_male | ✅ mai_female |
| Magahi        | mag  | ✅ mag_male | ✅ mag_female |
| English       | en   | ✅ en_male  | ✅ en_female  |

## 🐍 Python API

```python
from src.engine import TTSEngine

# Initialize engine
engine = TTSEngine(device="auto")  # auto-detects CPU/GPU/MPS

# Synthesize
output = engine.synthesize(
    text="गर्भावस्था में स्वस्थ आहार बहुत महत्वपूर्ण है",
    voice="hi_female",
    speed=1.0
)

# Save to file
engine.synthesize_to_file(
    text="नमस्ते",
    output_path="hello.wav",
    voice="hi_male"
)

# Get available voices
voices = engine.get_available_voices()
```

## 📁 Project Structure

```
TTS/
├── src/
│   ├── __init__.py
│   ├── config.py      # Language/voice configurations
│   ├── tokenizer.py   # Text tokenization & normalization
│   ├── engine.py      # Main TTS engine
│   ├── downloader.py  # HuggingFace model downloader
│   ├── api.py         # FastAPI REST server
│   └── cli.py         # Command-line interface
├── models/            # Downloaded models (created automatically)
├── dataset/           # SPICOR dataset (for fine-tuning)
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Environment variables:

- `MODELS_DIR`: Custom path for downloaded models
- `TTS_DEVICE`: Force device (cpu/cuda/mps)

## 📊 Performance

| Metric         | Value                              |
| -------------- | ---------------------------------- |
| Model Size     | ~300MB per voice                   |
| Inference Time | ~0.3s for short sentences (M2 Mac) |
| Sample Rate    | 22050 Hz                           |
| Audio Format   | 16-bit PCM WAV                     |

## 🙏 Credits

- **Models**: [SYSPIN](https://huggingface.co/SYSPIN) - IISc Bangalore
- **Architecture**: VITS (Conditional Variational Autoencoder with Adversarial Learning)
- **Framework**: [Coqui TTS](https://github.com/coqui-ai/TTS)
- **Dataset**: SPICOR TTS Project, IISc SPIRE Lab

## 📜 License

CC BY 4.0 - Same as the SYSPIN models

## 🤝 Contributing

This is a hackathon project. Feel free to fork and extend!

---

Built with ❤️ for Voice Tech for All Hackathon
