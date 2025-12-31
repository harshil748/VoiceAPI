# VoiceAPI: Multi-lingual Text-to-Speech for Healthcare

A production-ready, multi-lingual Text-to-Speech system supporting **11 Indian languages** with **21 voice variants**, trained on 150+ hours of speech data. Built for maternal healthcare accessibility.

🌐 **Live API**: [https://harshil748-voiceapi.hf.space](https://harshil748-voiceapi.hf.space)  
📖 **API Docs**: [https://harshil748-voiceapi.hf.space/docs](https://harshil748-voiceapi.hf.space/docs)  
💻 **GitHub**: [https://github.com/harshil748/VoiceAPI](https://github.com/harshil748/VoiceAPI)

---

## 🎯 Project Overview

Built for the **Voice Tech for All Hackathon** to address linguistic barriers in rural Indian healthcare. The system converts medical instructions into natural speech across 11 languages, enabling accessible prenatal care guidance for non-literate populations.

## 🏗️ System Architecture

### Overall System Design

![System Architecture](diagrams/system_architecture.png)

### Data Flow

![Data Flow](diagrams/data_flow.png)

### VITS Model Architecture

![Model Architecture](diagrams/model_architecture.png)

### Training Pipeline

![Training Pipeline](diagrams/training_pipeline.png)

### Supported Languages & Voices

![Voice Map](diagrams/voice_map.png)

## ✨ Key Features

- 🌏 **11 Indian Languages**: Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Chhattisgarhi, Maithili, Magahi, English, Gujarati
- 🎤 **21 Voice Variants**: Male & Female voices trained on 150+ hours of speech data
- 🎭 **Prosody Control**: 9 style presets (calm, happy, sad, slow, fast, etc.)
- ⚡ **Real-time Performance**: 0.3-0.9s inference on CPU hardware
- 🔌 **Production REST API**: FastAPI with automatic docs, CORS support
- 🧠 **Neural Architecture**: VITS + Meta MMS models with JIT optimization
- 📦 **Deployed on HuggingFace Spaces**: Always-on, cloud-hosted API

---

## 🚀 Try It Now (No Installation Required)

### Test with Python

```python
import requests

# Use the live API
base_url = 'https://harshil748-voiceapi.hf.space/Get_Inference'

params = {
    'text': 'नमस्ते, आप कैसे हैं?',  # Hindi text
    'lang': 'hindi',
}

# Upload any WAV file as speaker reference
with open('reference.wav', 'rb') as audio:
    response = requests.get(base_url, params=params, files={'speaker_wav': audio})

if response.status_code == 200:
    with open('output.wav', 'wb') as f:
        f.write(response.content)
    print("✅ Audio saved as 'output.wav'")
```

### Test with cURL

```bash
curl -X GET "https://harshil748-voiceapi.hf.space/Get_Inference?text=નમસ્તે&lang=gujarati" \
  -F "speaker_wav=@reference.wav" \
  -o output.wav
```

### Test with Postman

1. **Method**: `GET`
2. **URL**: `https://harshil748-voiceapi.hf.space/Get_Inference`
3. **Params Tab**:
   - `text`: Your text in any supported language
   - `lang`: One of: hindi, bengali, marathi, telugu, kannada, gujarati, bhojpuri, chhattisgarhi, maithili, magahi, english
4. **Body Tab** → `form-data`:
   - Key: `speaker_wav` (Type: File)
   - Value: Upload any `.wav` file
5. **Send** → Save response as `.wav` file

---

## 🎨 Supported Languages

| Language      | Code            | Male Voice | Female Voice | Sample Text                |
| ------------- | --------------- | ---------- | ------------ | -------------------------- |
| Hindi         | `hindi`         | ✅         | ✅           | नमस्ते                     |
| Bengali       | `bengali`       | ✅         | ✅           | নমস্কার                    |
| Marathi       | `marathi`       | ✅         | ✅           | नमस्कार                    |
| Telugu        | `telugu`        | ✅         | ✅           | నమస్కారం                   |
| Kannada       | `kannada`       | ✅         | ✅           | ನಮಸ್ಕಾರ                    |
| Gujarati      | `gujarati`      | ✅         | -            | નમસ્તે                     |
| Bhojpuri      | `bhojpuri`      | ✅         | ✅           | प्रणाम                     |
| Chhattisgarhi | `chhattisgarhi` | ✅         | ✅           | नमस्कार                    |
| Maithili      | `maithili`      | ✅         | ✅           | प्रणाम                     |
| Magahi        | `magahi`        | ✅         | ✅           | प्रणाम                     |
| English       | `english`       | ✅         | ✅           | hello (lowercase required) |

---

---

## 📡 API Reference

### GET /Get_Inference (Official Hackathon Endpoint)

Converts text to speech in any supported Indian language.

**Endpoint**: `https://harshil748-voiceapi.hf.space/Get_Inference`

**Parameters**:

| Parameter     | Type   | Required | Description                                           |
| ------------- | ------ | -------- | ----------------------------------------------------- |
| `text`        | string | ✅       | Text to convert to speech (English must be lowercase) |
| `lang`        | string | ✅       | Language code (see table above)                       |
| `speaker_wav` | file   | ✅       | Reference WAV file for speaker voice cloning          |

**Response**: `audio/wav` file (200 OK)

**Example**:

```python
import requests

response = requests.get(
    'https://harshil748-voiceapi.hf.space/Get_Inference',
    params={'text': 'ನಮಸ್ಕಾರ', 'lang': 'kannada'},
    files={'speaker_wav': open('reference.wav', 'rb')}
)

with open('output.wav', 'wb') as f:
    f.write(response.content)
```

---

## 📊 Technical Specifications

| Metric             | Value                                        |
| ------------------ | -------------------------------------------- |
| **Languages**      | 11 Indian languages                          |
| **Voice Variants** | 21 (male/female per language)                |
| **Training Data**  | 150+ hours (OpenSLR, Common Voice, IndicTTS) |
| **Model Size**     | 318MB (VITS), 998MB (Coqui)                  |
| **Inference Time** | 0.3-0.9 seconds per utterance                |
| **Sample Rate**    | 22.05kHz (VITS), 16kHz (MMS)                 |
| **Architecture**   | VITS + Meta MMS + Coqui TTS                  |
| **Deployment**     | HuggingFace Spaces (Docker)                  |
| **API Framework**  | FastAPI with Uvicorn                         |

---

## 🛠️ Local Development

### Installation

```bash
git clone https://github.com/harshil748/VoiceAPI
cd VoiceAPI

python3 -m venv tts
source tts/bin/activate  # On Windows: tts\Scripts\activate

pip install -r requirements.txt
```

### Start Local Server

```bash
python -m src.cli serve --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### Generate Speech Locally

```bash
python -m src.cli synthesize \
  --text "नमस्ते दोस्तों" \
  --voice hi_male \
  --output hello.wav

afplay hello.wav  # macOS
```

---

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
