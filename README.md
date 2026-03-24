# VoiceAPI — Multi-lingual TTS + Voice Cloning (Local)

A fully **local** multi-lingual Text-to-Speech system supporting **11 Indian languages**
with **21 voice variants** and real-time **voice cloning** — no cloud APIs, no API keys,
no internet connection required after first setup.

> **All inference runs on your machine** using model weights stored in `models/`.  
> Voice cloning uses [Coqui XTTS v2](https://github.com/coqui-ai/TTS) (downloaded once, cached locally).

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🌏 **11 Languages** | Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Chhattisgarhi, Maithili, Magahi, English, Gujarati |
| 🎤 **21 Voice Variants** | Male & Female per language (SYSPIN VITS models) |
| 🧬 **Voice Cloning** | Upload any 5–30 s WAV → synthesise in that voice via XTTS v2 |
| 🎭 **Prosody Control** | Speed · Pitch · Energy sliders + 9 style presets |
| ⚡ **Fast Inference** | 0.3–0.9 s per utterance on CPU |
| 🖥️ **Web UI** | Next.js frontend — language picker, clone mode, audio playback + WAV download |
| 🔌 **REST API** | FastAPI with auto-generated `/docs` (Swagger UI) |
| 📴 **Fully Offline** | After first model download everything runs without internet |

---

## 🚀 Quick Start

### 1 — Clone & install Python deps

```bash
git clone https://github.com/harshil748/VoiceAPI
cd VoiceAPI

# Create a virtual environment (recommended)
python3 -m venv tts
source tts/bin/activate        # Windows: tts\Scripts\activate

pip install -r requirements.txt
```

> **GPU users**: swap the `torch` line in `requirements.txt` for the CUDA wheel from
> [pytorch.org](https://pytorch.org/get-started/locally/) before installing.

---

### 2 — Start everything with one command

```bash
chmod +x start.sh
./start.sh
```

This script will:
1. Check Python + Node dependencies (install missing ones automatically)
2. Start the **FastAPI backend** on `http://localhost:8000`
3. Wait for the API to be healthy
4. Start the **Next.js web UI** on `http://localhost:3000`
5. Print a summary and keep both processes alive (Ctrl+C stops both)

| Service | URL |
|---|---|
| Web UI | http://localhost:3000 |
| API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

### 3 — Or start services individually

**API server only:**
```bash
python start_api.py                        # default: 0.0.0.0:8000
python start_api.py --port 8001            # custom port
python start_api.py --reload               # hot-reload (dev mode)
python start_api.py --preload hi_female    # preload a voice at startup
```

**Web UI only** (assumes API is already running):
```bash
cd web
npm install          # first time only
npm run dev
```

---

## 🎙️ Voice Cloning

Voice cloning works for: **English, Hindi, Bengali, Gujarati, Marathi, Telugu, Kannada**.

### Via the Web UI
1. Open `http://localhost:3000`
2. Select **"Custom Voice Clone"** mode
3. Choose language and style
4. Upload a `.wav` file (5–30 s of clean speech)
5. Click **Generate Audio** → play or download the `.wav`

### Via the API (Python)
```python
import requests

with open("my_voice.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/clone",
        params={
            "text": "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?",
            "lang": "hindi",
            "style": "calm",
            "speed": 1.0,
        },
        files={"speaker_wav": f},
    )

with open("output.wav", "wb") as out:
    out.write(response.content)
print("Saved output.wav")
```

### Via cURL
```bash
curl -X POST "http://localhost:8000/clone?text=Hello+world&lang=english&style=default" \
     -F "speaker_wav=@my_voice.wav" \
     -o cloned_output.wav
```

> **First clone request**: XTTS v2 weights (~1.8 GB) are downloaded from HuggingFace and
> cached in `models/` automatically. All subsequent requests are fully offline.

---

## 🗣️ Standard Synthesis (no reference audio needed)

```python
import requests

response = requests.post(
    "http://localhost:8000/synthesize",
    json={
        "text": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        "voice": "kn_female",
        "style": "happy",
        "speed": 1.0,
        "pitch": 1.0,
        "energy": 1.0,
    },
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

Or use the GET convenience endpoint in a browser:
```
http://localhost:8000/synthesize/get?text=Hello&voice=en_female&style=calm
```

---

## 📡 API Reference

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Server + engine status |
| `GET` | `/voices` | All voices with download/load status |
| `GET` | `/styles` | Style presets and parameter descriptions |
| `GET` | `/languages` | Supported language codes |
| `POST` | `/synthesize` | Synthesise text → WAV (JSON body) |
| `GET` | `/synthesize/get` | Synthesise text → WAV (query params) |
| `POST` | `/synthesize/stream` | Streaming WAV response |
| `POST` | `/clone` | **Voice cloning** via XTTS v2 (multipart) |
| `GET\|POST` | `/Get_Inference` | Hackathon-spec endpoint (clones when possible) |
| `POST` | `/preload` | Load a voice into memory |
| `POST` | `/unload` | Unload a voice from memory |
| `POST` | `/batch` | Batch synthesise multiple texts |

Full interactive docs: **http://localhost:8000/docs**

---

### POST /clone — Voice Cloning

| Parameter | Type | Required | Description |
|---|---|---|---|
| `text` | string (query) | ✅ | Text to synthesise |
| `lang` | string (query) | ✅ | `english`, `hindi`, `bengali`, `gujarati`, `marathi`, `telugu`, `kannada` |
| `speaker_wav` | file (form) | ✅ | Reference audio — WAV or MP3, 5–30 s recommended |
| `style` | string (query) | ❌ | `default`, `calm`, `happy`, `sad`, `slow`, `fast`, `soft`, `loud`, `excited` |
| `speed` | float (query) | ❌ | 0.5 – 2.0 (default 1.0) |
| `pitch` | float (query) | ❌ | 0.5 – 2.0 (default 1.0) |
| `energy` | float (query) | ❌ | 0.5 – 2.0 (default 1.0) |

**Response**: `audio/wav` · Headers include `X-Duration`, `X-Sample-Rate`, `X-Inference-Time`

---

### POST /synthesize — Standard TTS

```json
{
  "text": "নমস্কার, আপনি কেমন আছেন?",
  "voice": "bn_female",
  "speed": 1.0,
  "pitch": 1.0,
  "energy": 1.0,
  "style": "calm",
  "normalize": true
}
```

---

### GET|POST /Get_Inference — Hackathon Spec

```bash
# With voice cloning (XTTS-supported language)
curl -G "http://localhost:8000/Get_Inference" \
     --data-urlencode "text=नमस्ते" \
     --data-urlencode "lang=hindi" \
     -F "speaker_wav=@reference.wav" \
     -o output.wav

# Non-XTTS language (uses pre-trained VITS, speaker_wav accepted but not used)
curl -G "http://localhost:8000/Get_Inference" \
     --data-urlencode "text=का बा?" \
     --data-urlencode "lang=bhojpuri" \
     -F "speaker_wav=@reference.wav" \
     -o output.wav
```

---

## 🌐 Supported Languages

| Language | Voice Keys | Clone Support | Notes |
|---|---|---|---|
| Hindi | `hi_male`, `hi_female` | ✅ XTTS | SYSPIN VITS JIT |
| Bengali | `bn_male`, `bn_female` | ✅ XTTS | SYSPIN VITS JIT |
| Marathi | `mr_male`, `mr_female` | ✅ XTTS | SYSPIN VITS JIT |
| Telugu | `te_male`, `te_female` | ✅ XTTS | SYSPIN VITS JIT |
| Kannada | `kn_male`, `kn_female` | ✅ XTTS | SYSPIN VITS JIT |
| English | `en_male`, `en_female` | ✅ XTTS | text must be lowercase |
| Gujarati | `gu_mms` | ✅ XTTS | Facebook MMS (auto-downloads) |
| Bhojpuri | `bho_male`, `bho_female` | ❌ (VITS only) | Coqui .pth checkpoint |
| Chhattisgarhi | `hne_male`, `hne_female` | ❌ (VITS only) | SYSPIN VITS JIT |
| Maithili | `mai_male`, `mai_female` | ❌ (VITS only) | SYSPIN VITS JIT |
| Magahi | `mag_male`, `mag_female` | ❌ (VITS only) | SYSPIN VITS JIT |

---

## 🛠️ CLI Reference

```bash
# List all voices and download status
python -m src.cli list

# Download a specific voice
python -m src.cli download --voice hi_male

# Download all voices for a language
python -m src.cli download --lang bn

# Download everything
python -m src.cli download --all

# Synthesise from the command line
python -m src.cli synthesize \
  --text "नमस्ते दोस्तों" \
  --voice hi_female \
  --output hello.wav

# Start API server (equivalent to start_api.py)
python -m src.cli serve --port 8000 --reload
```

---

## 📁 Repository Structure

```
VoiceAPI/
├── src/
│   ├── api.py           # FastAPI REST server (local, no cloud deps)
│   ├── engine.py        # Unified TTS inference engine
│   ├── tokenizer.py     # Indic script tokenisation (VITS-compatible)
│   ├── config.py        # Language / voice / style configurations
│   ├── downloader.py    # HuggingFace model downloader
│   └── cli.py           # Command-line interface
│
├── models/              # All model weights (local)
│   ├── hi_female/       # hi_female_vits_30hrs.pt + chars.txt
│   ├── bn_female/       # bn_female_vits_30hrs.pt + chars.txt
│   ├── bho_female/      # checkpoint_340000.pth + config.json
│   ├── gu_mms/          # Facebook MMS tokeniser (weights auto-downloaded)
│   ├── ...              # (all 20 SYSPIN voices)
│   └── tts_models--multilingual--multi-dataset--xtts_v2/
│                        # XTTS v2 weights (auto-downloaded on first clone)
│
├── web/                 # Next.js frontend
│   ├── app/
│   │   ├── page.js      # Main UI (clone + standard synthesis modes)
│   │   ├── layout.js
│   │   └── globals.css
│   ├── .env.local       # NEXT_PUBLIC_API_BASE=http://localhost:8000
│   └── package.json
│
├── local_tests/         # Integration test suite
├── training/            # Training scripts (VITS fine-tuning)
│
├── start.sh             # One-command launcher (API + Web UI)
├── start_api.py         # API-only launcher with CLI flags
└── requirements.txt     # Python dependencies
```

---

## ⚙️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js Web UI (:3000)                   │
│   Clone Mode ──► POST /clone         (XTTS v2 voice clone)  │
│   Standard  ──► POST /synthesize     (VITS / Coqui / MMS)   │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP (localhost)
┌──────────────────────────▼──────────────────────────────────┐
│                  FastAPI Server (:8000)                       │
│  src/api.py                                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    TTSEngine  (src/engine.py)                 │
│                                                               │
│   JIT .pt models    Coqui .pth models   Facebook MMS          │
│   (19 SYSPIN        (Bhojpuri via       (Gujarati via          │
│    voices)           TTS.Synthesizer)    transformers)         │
│                                                               │
│   XTTS v2  ──────────────────────────────────────────────►   │
│   (voice cloning — weights cached in models/ after 1st use)  │
└──────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      models/  (local disk)                    │
│   SYSPIN VITS .pt  ·  Bhojpuri .pth  ·  MMS config           │
│   XTTS v2 weights  (~1.8 GB, downloaded once)                 │
└──────────────────────────────────────────────────────────────┘
```

### Model Types

| Type | Format | Loader | Languages |
|---|---|---|---|
| SYSPIN VITS JIT | `.pt` + `chars.txt` | `torch.jit.load` | Hindi, Bengali, Marathi, Telugu, Kannada, English, Chhattisgarhi, Maithili, Magahi |
| Coqui Checkpoint | `.pth` + `config.json` | `TTS.Synthesizer` | Bhojpuri |
| Facebook MMS | HF `VitsModel` | `transformers` | Gujarati |
| XTTS v2 | HF cached weights | `TTS.api.TTS` | Voice cloning (EN, HI, BN, GU, MR, TE, KN) |

---

## 🔧 Configuration

### Change API port

```bash
# Via environment variable
API_PORT=8001 WEB_PORT=3001 ./start.sh

# Or directly
python start_api.py --port 8001
```

Then update `web/.env.local`:
```
NEXT_PUBLIC_API_BASE=http://localhost:8001
```

### Preload voices at startup (faster first request)

```bash
python start_api.py --preload hi_female en_female bn_female
```

### GPU inference (CUDA)

Install the CUDA PyTorch wheel and the engine will auto-detect the GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
python start_api.py  # device: cuda
```

---

## 🧪 Running Tests

```bash
# Make sure the API is running first
python start_api.py &

# Run the local integration test suite
cd local_tests
python test_local_api.py

# Test voice cloning across all supported languages
python test_live_clone_all_languages.py
```

---

## 📊 Performance

| Metric | Value |
|---|---|
| Languages | 11 Indian languages |
| Voice variants | 21 (male + female) |
| Inference time | 0.3–0.9 s per utterance (CPU) |
| Sample rate | 22 050 Hz (VITS), 16 000 Hz (MMS), 24 000 Hz (XTTS) |
| XTTS first load | ~15–30 s (subsequent: ~5 s) |
| XTTS model size | ~1.8 GB (downloaded once, cached in `models/`) |
| SYSPIN model size | ~320 MB per voice |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `torch` | Neural network inference |
| `TTS` | Coqui TTS — Bhojpuri checkpoints + XTTS v2 voice cloning |
| `transformers` | Facebook MMS Gujarati model |
| `huggingface-hub` | Model snapshot downloads |
| `soundfile` | WAV I/O |
| `librosa` | Pitch shift + time stretch |
| `fastapi` + `uvicorn` | REST API server |
| `next` (Node) | Web UI |

---

## 🙏 Acknowledgments

- **SYSPIN** — VITS model weights for 10 Indian languages
- **Meta AI** — MMS multilingual speech model (Gujarati)
- **Coqui TTS** — XTTS v2 multilingual voice cloning
- **OpenSLR / Common Voice / IndicTTS** — Training datasets

---

## 📜 License

- **Code**: MIT License
- **SYSPIN Models**: CC BY 4.0
- **MMS Models**: CC BY-NC 4.0
- **XTTS v2**: Coqui Public Model License

---

## 👥 Team

Built by **Team VoiceAPI** — CHARUSAT University

- **Harshil Patel**
- **Harnish Patel**
- **Aman Paya**