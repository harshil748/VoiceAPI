# Voice Tech for All: Technical Report

## Multi-lingual Text-to-Speech System with Style Transfer

**Hackathon**: Voice Tech for All  
**Date**: December 2025

---

## Executive Summary

We present a **multi-lingual Text-to-Speech (TTS) system** supporting **11 Indian languages** with **style/prosody control** capabilities. The system is designed for deployment as a healthcare assistant for pregnant mothers in low-income communities, making health information accessible in native languages.

### Key Achievements

| Metric                 | Value                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| Languages Supported    | 11 (Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Chhattisgarhi, Maithili, Magahi, English, Gujarati) |
| Voice Variants         | 21 (male + female for each language)                                                                        |
| Style Presets          | 9 (default, slow, fast, soft, loud, happy, sad, calm, excited)                                              |
| Average Inference Time | ~0.3s (CPU, Apple M2)                                                                                       |
| Model Size             | ~300MB per voice (VITS), ~145MB (MMS)                                                                       |
| API Latency            | <500ms for typical sentences                                                                                |

---

## 1. System Architecture

### 1.1 Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    REST API Server (FastAPI)                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────────┐│
│  │/synthesize│  │ /voices     │  │ /styles               ││
│  │ /stream   │  │ /languages  │  │ /health               ││
│  └──────────┘  └──────────────┘  └─────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                      TTS Engine                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Text Normalizer │→ │ Tokenizer       │→ │ VITS/MMS    │ │
│  │ (Indian scripts)│  │ (char-to-ID)    │  │ Inference   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│                              ↓                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Style Processor (Prosody Control)          ││
│  │  • Pitch Shifting (librosa)                             ││
│  │  • Time Stretching (speed control)                      ││
│  │  • Energy/Volume Modification                           ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    Model Repository                          │
│  ┌────────────────────┐  ┌────────────────────────────────┐ │
│  │ SYSPIN VITS Models │  │ Facebook MMS Models            │ │
│  │ (10 languages)     │  │ (Gujarati)                     │ │
│  └────────────────────┘  └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Component Details

#### Text Normalizer

- Handles Indian script peculiarities
- Converts number notations: `{100}{एकसो}` → `एकसो`
- Normalizes punctuation across scripts
- Handles code-switching (Hindi in English text)

#### VITS Models (SYSPIN)

- **Architecture**: Conditional Variational Autoencoder with Adversarial Learning
- **Training Data**: 20-30 hours per speaker from IISc Bangalore
- **Output**: 22050 Hz, 16-bit PCM
- **Languages**: Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Chhattisgarhi, Maithili, Magahi, English

#### MMS Model (Facebook)

- **Architecture**: VITS-based, trained on MMS corpus
- **Output**: 16000 Hz
- **Languages**: Gujarati (and 1100+ others available)
- **Model Size**: 145MB

#### Style Processor

- **Pitch Shifting**: Using librosa phase vocoder
- **Time Stretching**: WSOLA algorithm via librosa
- **Energy Control**: Soft clipping with tanh for natural sound

---

## 2. API Specification

### 2.1 Endpoints

| Endpoint             | Method | Description                      |
| -------------------- | ------ | -------------------------------- |
| `/`                  | GET    | API info and documentation links |
| `/health`            | GET    | System health and loaded models  |
| `/voices`            | GET    | List all available voices        |
| `/languages`         | GET    | List supported languages         |
| `/styles`            | GET    | List style presets               |
| `/synthesize`        | POST   | Generate speech from text        |
| `/synthesize/get`    | GET    | Simple synthesis (for testing)   |
| `/synthesize/stream` | POST   | Streaming audio response         |
| `/preload`           | POST   | Preload voice into memory        |
| `/batch`             | POST   | Batch synthesis                  |

### 2.2 Synthesis Request

```json
{
	"text": "નમસ્તે, હું તમારી કેવી રીતે મદદ કરી શકું?",
	"voice": "gu_mms",
	"speed": 1.0,
	"pitch": 1.0,
	"energy": 1.0,
	"style": "calm",
	"normalize": true
}
```

### 2.3 Style Presets

| Preset  | Speed | Pitch | Energy | Use Case               |
| ------- | ----- | ----- | ------ | ---------------------- |
| default | 1.0   | 1.0   | 1.0    | Normal speech          |
| slow    | 0.75  | 1.0   | 1.0    | Elderly users, clarity |
| fast    | 1.25  | 1.0   | 1.0    | Quick information      |
| soft    | 0.9   | 0.95  | 0.7    | Calming content        |
| loud    | 1.0   | 1.05  | 1.3    | Alerts, emphasis       |
| happy   | 1.1   | 1.1   | 1.2    | Positive messages      |
| sad     | 0.85  | 0.9   | 0.8    | Empathetic responses   |
| calm    | 0.9   | 0.95  | 0.85   | Healthcare guidance    |
| excited | 1.2   | 1.15  | 1.3    | Celebrations           |

---

## 3. Supported Languages

| Language      | Code | Voices       | Model Type   | Sample Rate |
| ------------- | ---- | ------------ | ------------ | ----------- |
| Hindi         | hi   | Male, Female | SYSPIN VITS  | 22050 Hz    |
| Bengali       | bn   | Male, Female | SYSPIN VITS  | 22050 Hz    |
| Marathi       | mr   | Male, Female | SYSPIN VITS  | 22050 Hz    |
| Telugu        | te   | Male, Female | SYSPIN VITS  | 22050 Hz    |
| Kannada       | kn   | Male, Female | SYSPIN VITS  | 22050 Hz    |
| Bhojpuri      | bho  | Male, Female | SYSPIN VITS  | 22050 Hz    |
| Chhattisgarhi | hne  | Male, Female | SYSPIN VITS  | 22050 Hz    |
| Maithili      | mai  | Male, Female | SYSPIN VITS  | 22050 Hz    |
| Magahi        | mag  | Male, Female | SYSPIN VITS  | 22050 Hz    |
| English       | en   | Male, Female | SYSPIN VITS  | 22050 Hz    |
| Gujarati      | gu   | Neutral      | Facebook MMS | 16000 Hz    |

---

## 4. Implementation Details

### 4.1 Technology Stack

| Component         | Technology                               |
| ----------------- | ---------------------------------------- |
| Backend Framework | FastAPI                                  |
| ML Framework      | PyTorch                                  |
| TTS Models        | VITS (Coqui AI / SYSPIN), MMS (Facebook) |
| Audio Processing  | librosa, soundfile, scipy                |
| Model Hub         | Hugging Face Hub                         |
| API Documentation | OpenAPI/Swagger                          |

### 4.2 Model Architecture - VITS

VITS (Conditional Variational Autoencoder with Adversarial Learning) was chosen for:

- **End-to-End Efficiency**: Combines acoustic modeling and vocoding in a single pass
- **High Quality**: Natural-sounding speech comparable to two-stage systems
- **Multi-Speaker Support**: Supports different speakers via embeddings
- **Fast Inference**: TorchScript JIT compilation for speed

### 4.3 Style/Accent Transfer Implementation

Our style transfer uses **post-processing** approach for simplicity and reliability:

1. **Pitch Shifting**: Phase vocoder via librosa

   ```python
   semitones = 12 * np.log2(pitch_factor)
   shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
   ```

2. **Time Stretching**: WSOLA algorithm

   ```python
   stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
   ```

3. **Energy Control**: Soft clipping for natural sound
   ```python
   modified = audio * energy_factor
   if energy_factor > 1.0:
       modified = np.tanh(modified * 2) * 0.95  # Soft clip
   ```

### 4.4 Key Design Decisions

1. **TorchScript Models**: JIT-compiled for faster inference
2. **Lazy Loading**: Models loaded on-demand to minimize memory
3. **CPU Fallback**: Apple Silicon MPS compatibility issues handled
4. **Streaming Support**: Progressive audio delivery for real-time apps

---

## 5. Usage Examples

### 5.1 Python API

```python
from src.engine import TTSEngine

# Initialize engine
engine = TTSEngine(device="auto")

# Basic synthesis
output = engine.synthesize(
    text="गर्भावस्था में स्वस्थ आहार बहुत महत्वपूर्ण है",
    voice="hi_female"
)

# With style control
output = engine.synthesize(
    text="आपका दिन शुभ हो",
    voice="hi_male",
    style="happy",
    pitch=1.1
)

# Gujarati
output = engine.synthesize(
    text="સ્વસ્થ રહો, ખુશ રહો",
    voice="gu_mms",
    style="calm"
)
```

### 5.2 REST API

```bash
# Basic synthesis
curl -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते", "voice": "hi_male"}' \
  --output speech.wav

# With style
curl -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "आपका स्वागत है", "voice": "hi_female", "style": "happy"}' \
  --output welcome.wav

# Gujarati
curl -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "નમસ્તે", "voice": "gu_mms"}' \
  --output gujarati.wav
```

### 5.3 Command Line

```bash
# Download models
python -m src.cli download --voice hi_male
python -m src.cli download --lang hi  # All Hindi voices

# Synthesize
python -m src.cli synthesize --text "नमस्ते" --voice hi_male --output hello.wav

# Start server
python -m src.cli serve --port 8000
```

---

## 6. Healthcare Use Case

### 6.1 Target Application

The TTS system is designed for integration with an **LLM-based healthcare assistant** for pregnant mothers in low-income communities.

### 6.2 Key Features for Healthcare

1. **Multi-lingual Support**: Information in native languages
2. **Calm Style Preset**: Reassuring tone for medical guidance
3. **Slow Speed Option**: Clear pronunciation for instructions
4. **Low Latency**: Real-time conversational responses

### 6.3 Example Healthcare Dialogue

```
User: "ગર્ભાવસ્થામાં શું ખાવું જોઈએ?"

System Response (TTS with calm style in Gujarati):
"ગર્ભાવસ્થામાં તમારે પ્રોટીન, આયર્ન અને ફોલિક એસિડથી ભરપૂર
ખોરાક લેવો જોઈએ. દાળ, પાલક, ઈંડા અને દૂધ સારા વિકલ્પો છે."
```

---

## 7. Performance Benchmarks

| Test                    | Time  | Notes                              |
| ----------------------- | ----- | ---------------------------------- |
| Hindi synthesis (short) | 0.25s | "नमस्ते"                           |
| Hindi synthesis (long)  | 0.45s | 50-word sentence                   |
| Gujarati MMS            | 0.35s | First load includes model download |
| Style processing        | +0.1s | Pitch + speed adjustment           |
| API round-trip          | 0.5s  | Including network overhead         |

Hardware: Apple M2 Pro, 16GB RAM, CPU inference

---

## 8. Deployment

### 8.1 Quick Start

```bash
# Clone repository
git clone https://github.com/harshil748/VoiceAPI
cd VoiceAPI

# Setup environment
python3 -m venv tts
source tts/bin/activate
pip install -r requirements.txt

# Download a model
python -m src.cli download --voice hi_male

# Start server
python -m src.cli serve --port 8000
```

### 8.2 Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python -m src.cli download --lang hi
EXPOSE 8000
CMD ["python", "-m", "src.cli", "serve"]
```

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

1. **Model Size**: Each VITS model is ~300MB
2. **MPS Compatibility**: Apple Silicon MPS not fully supported
3. **Real-time Streaming**: Limited to sentence-level
4. **Gujarati Gender**: MMS has only neutral voice

### 9.2 Future Improvements

1. **Model Quantization**: INT8 for smaller size
2. **Voice Cloning**: Reference audio-based synthesis
3. **SSML Support**: Markup language for fine control
4. **More Languages**: Odia, Assamese, Punjabi
5. **Fine-tuning**: Custom voice training on SPICOR data

---

## 10. Credits

### Model Sources

| Source                  | Models                | License      |
| ----------------------- | --------------------- | ------------ |
| SYSPIN (IISc Bangalore) | VITS for 10 languages | CC BY 4.0    |
| Facebook MMS            | Gujarati VITS         | CC BY-NC 4.0 |

### Dataset

- **SPICOR TTS Project**: IISc SPIRE Lab, Bangalore
- **Audio Quality**: 48kHz, 24-bit, mono

### Frameworks

- Coqui TTS, Hugging Face Transformers, FastAPI, librosa

---

## 11. Conclusion

We have developed a comprehensive multi-lingual TTS system that:

✅ Supports **11 Indian languages** with 21 voice variants  
✅ Provides **9 style presets** for prosody control  
✅ Offers a **REST API** with OpenAPI documentation  
✅ Achieves **<500ms latency** for typical sentences  
✅ Is **production-ready** with proper error handling

The system is well-suited for the healthcare assistant use case, providing clear, natural-sounding speech in native languages to help pregnant mothers access healthcare information.

---

**Repository**: https://github.com/harshil748/VoiceAPI  
**API Documentation**: http://localhost:8000/docs
