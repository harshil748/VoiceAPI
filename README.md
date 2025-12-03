# Voice Tech for All - Multi-lingual TTS System

A lightweight, multi-lingual Text-to-Speech system supporting **11 Indian languages** with **style/prosody control** and REST API.

## 🎯 Hackathon: Voice Tech for All

Built for the healthcare assistant use case - helping pregnant mothers in low-income communities access healthcare information in their native languages.

## ✨ Features

- **11 Indian Languages**: Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Chhattisgarhi, Maithili, Magahi, English, **Gujarati**
- **21 Voice Options**: Male & Female voices for each language
- **Style/Prosody Control**: 9 presets (happy, sad, calm, excited, etc.)
- **Pitch & Speed Control**: Fine-tune voice characteristics
- **Lightweight**: VITS-based models optimized for fast inference
- **REST API**: FastAPI-powered server with OpenAPI docs
- **Text Normalization**: Handles numbers, punctuation for Indian scripts

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and navigate
git clone https://github.com/harshil748/VoiceAPI
cd VoiceAPI

# Create virtual environment
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

# Gujarati uses Facebook MMS (auto-downloads on first use)
```

### 3. Synthesize Speech

```bash
# Basic synthesis
python -m src.cli synthesize --text "नमस्ते दोस्तों" --voice hi_male --output hello.wav

# Play the audio (macOS)
afplay hello.wav
```

### 4. Start API Server

```bash
python -m src.cli serve --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## 🎨 Style Presets

| Preset    | Speed | Pitch | Energy | Best For                |
| --------- | ----- | ----- | ------ | ----------------------- |
| `default` | 1.0   | 1.0   | 1.0    | Normal speech           |
| `slow`    | 0.75  | 1.0   | 1.0    | Elderly users, clarity  |
| `fast`    | 1.25  | 1.0   | 1.0    | Quick information       |
| `soft`    | 0.9   | 0.95  | 0.7    | Calming content         |
| `loud`    | 1.0   | 1.05  | 1.3    | Alerts, emphasis        |
| `happy`   | 1.1   | 1.1   | 1.2    | Positive messages       |
| `sad`     | 0.85  | 0.9   | 0.8    | Empathetic responses    |
| `calm`    | 0.9   | 0.95  | 0.85   | **Healthcare guidance** |
| `excited` | 1.2   | 1.15  | 1.3    | Celebrations            |

## 📡 API Usage

### 🏆 Hackathon API - GET /Get_Inference

**This is the official hackathon endpoint** that follows the Voice Tech for All specification:

```python
import requests

base_url = 'http://localhost:8000/Get_Inference'
WavPath = 'path/to/reference.wav'

params = {
    'text': 'ಮಾದರಿಯು ಸರಿಯಾಗಿ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತಿದೆಯೇ ಎಂದು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಲು ಬಳಸಲಾಗುವ ಪರೀಕ್ಷಾ ವಾಕ್ಯ ಇದು.',
    'lang': 'kannada',
}

with open(WavPath, "rb") as AudioFile:
    response = requests.get(base_url, params=params, files={'speaker_wav': AudioFile})

if response.status_code == 200:
    with open('output.wav', 'wb') as f:
        f.write(response.content)
    print("Audio saved as 'output.wav'")
```

**Query Parameters:**

| Parameter     | Type   | Required  | Description                                                                                                      |
| ------------- | ------ | --------- | ---------------------------------------------------------------------------------------------------------------- |
| `text`        | string | Mandatory | Input text to convert to speech. For English, text must be lowercase.                                            |
| `lang`        | string | Mandatory | Language: bhojpuri, bengali, english, gujarati, hindi, chhattisgarhi, kannada, magahi, maithili, marathi, telugu |
| `speaker_wav` | file   | Mandatory | Reference WAV file for speaker voice                                                                             |

**Response:** `200 OK` with `Content-Type: audio/wav`

---

### Synthesize with Style (POST)

```bash
curl -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "आपका दिन शुभ हो",
    "voice": "hi_female",
    "style": "happy",
    "speed": 1.0,
    "pitch": 1.0
  }' \
  --output speech.wav
```

### Gujarati Synthesis

```bash
curl -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "નમસ્તે, કેમ છો?", "voice": "gu_mms", "style": "calm"}' \
  --output gujarati.wav
```

### List Style Presets

```bash
curl http://localhost:8000/styles
```

## 🎤 Available Voices

| Language      | Code | Male        | Female        | Notes        |
| ------------- | ---- | ----------- | ------------- | ------------ |
| Hindi         | hi   | ✅ hi_male  | ✅ hi_female  | SYSPIN       |
| Bengali       | bn   | ✅ bn_male  | ✅ bn_female  | SYSPIN       |
| Marathi       | mr   | ✅ mr_male  | ✅ mr_female  | SYSPIN       |
| Telugu        | te   | ✅ te_male  | ✅ te_female  | SYSPIN       |
| Kannada       | kn   | ✅ kn_male  | ✅ kn_female  | SYSPIN       |
| Bhojpuri      | bho  | ✅ bho_male | ✅ bho_female | SYSPIN       |
| Chhattisgarhi | hne  | ✅ hne_male | ✅ hne_female | SYSPIN       |
| Maithili      | mai  | ✅ mai_male | ✅ mai_female | SYSPIN       |
| Magahi        | mag  | ✅ mag_male | ✅ mag_female | SYSPIN       |
| English       | en   | ✅ en_male  | ✅ en_female  | SYSPIN       |
| **Gujarati**  | gu   | ✅ gu_mms   | -             | Facebook MMS |

## 🐍 Python API

```python
from src.engine import TTSEngine

# Initialize engine
engine = TTSEngine(device="auto")

# Basic synthesis
output = engine.synthesize(
    text="गर्भावस्था में स्वस्थ आहार महत्वपूर्ण है",
    voice="hi_female"
)

# With style control
output = engine.synthesize(
    text="आपका दिन शुभ हो",
    voice="hi_male",
    style="happy",      # Use preset
    pitch=1.1,          # Or manual control
    speed=1.0,
    energy=1.2
)

# Gujarati
output = engine.synthesize(
    text="સ્વસ્થ રહો, ખુશ રહો",
    voice="gu_mms",
    style="calm"
)

# Save to file
engine.synthesize_to_file(
    text="નમસ્તે",
    output_path="hello.wav",
    voice="gu_mms",
    style="calm"
)
```

## 📁 Project Structure

```text
VoiceAPI/
├── src/
│   ├── config.py      # Language/voice/style configurations
│   ├── tokenizer.py   # Text tokenization & normalization
│   ├── engine.py      # Main TTS engine with style processor
│   ├── downloader.py  # HuggingFace model downloader
│   ├── api.py         # FastAPI REST server
│   └── cli.py         # Command-line interface
├── models/            # Downloaded models
├── dataset/           # SPICOR dataset (for fine-tuning)
├── technical_report.md
├── requirements.txt
└── README.md
```

## 📊 Performance

| Metric         | Value                           |
| -------------- | ------------------------------- |
| Languages      | 11                              |
| Voice Variants | 21                              |
| Style Presets  | 9                               |
| Model Size     | ~300MB (VITS), ~145MB (MMS)     |
| Inference Time | ~0.3s (M2 Mac, CPU)             |
| Sample Rate    | 22050 Hz (VITS), 16000 Hz (MMS) |

## 🙏 Credits

- **SYSPIN Models**: [IISc Bangalore](https://huggingface.co/SYSPIN)
- **MMS Models**: [Facebook Research](https://huggingface.co/facebook/mms-tts-guj)
- **Architecture**: VITS (Coqui AI)
- **Dataset**: SPICOR TTS Project, IISc SPIRE Lab

## 📜 License

CC BY 4.0 (SYSPIN), CC BY-NC 4.0 (MMS)

---

Built with ❤️ for **Voice Tech for All Hackathon**
