"""
Local REST API Server — Multi-lingual TTS + Voice Cloning
FastAPI server that runs 100 % locally using models in the project's
models/ directory.  No external API keys or cloud services required.

Endpoints:
  GET  /                    — welcome
  GET  /health              — server + engine status
  GET  /voices              — list all available voices
  GET  /styles              — list style presets
  GET  /languages           — list supported languages

  POST /synthesize          — standard TTS (JSON body)
  GET  /synthesize/get      — standard TTS (query params, handy for testing)
  POST /synthesize/stream   — streaming WAV response

  POST /clone               — voice cloning via XTTS v2 (multipart upload)

  GET|POST /Get_Inference   — hackathon-spec endpoint (multipart, uses local VITS)

  POST /preload             — preload a voice model into memory
  POST /unload              — unload a voice model from memory
  POST /batch               — batch synthesise multiple texts
"""

import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import (
    LANGUAGE_CONFIGS,
    STYLE_PRESETS,
    get_available_languages,
)
from .engine import TTSEngine, TTSOutput

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Languages supported by XTTS v2 for voice cloning
XTTS_LANG_MAP: Dict[str, str] = {
    "english": "en",
    "hindi": "hi",
    "bengali": "bn",
    "gujarati": "gu",
    "marathi": "mr",
    "telugu": "te",
    "kannada": "kn",
}

# Default voice per language for the hackathon /Get_Inference endpoint
LANG_TO_VOICE: Dict[str, str] = {
    "hindi": "hi_female",
    "bengali": "bn_female",
    "marathi": "mr_female",
    "telugu": "te_female",
    "kannada": "kn_female",
    "bhojpuri": "bho_female",
    "chhattisgarhi": "hne_female",
    "maithili": "mai_female",
    "magahi": "mag_female",
    "english": "en_female",
    "gujarati": "gu_female",
}

# Upload limits
MAX_UPLOAD_BYTES: int = int(
    os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))
)  # 20 MB

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VoiceAPI — Multi-lingual TTS (Local)",
    description="""
A fully local multi-lingual Text-to-Speech API.

## Features
- **Standard synthesis** — 11 Indian languages, male & female voices
- **Voice cloning** — clone any voice from a short WAV sample using XTTS v2
- **Style control** — speed, pitch, energy sliders + preset styles
- **Fully offline** — all model weights live in `models/`; no API keys needed

## Languages
Hindi · Bengali · Marathi · Telugu · Kannada · Bhojpuri · Chhattisgarhi ·
Maithili · Magahi · English · Gujarati
""",
    version="2.0.0",
    contact={"name": "Harshil Patel", "url": "https://harshilpatel.me/#contact"},
    license_info={
        "name": "CC BY 4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------

_engine: Optional[TTSEngine] = None


def get_engine() -> TTSEngine:
    """Return the shared TTSEngine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = TTSEngine(device="auto")
    return _engine


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SynthesizeRequest(BaseModel):
    text: str = Field(
        ..., description="Text to synthesize", min_length=1, max_length=5000
    )
    voice: str = Field(
        "hi_female", description="Voice key, e.g. hi_male, bn_female, gu_mms"
    )
    speed: float = Field(1.0, description="Speed multiplier (0.5–2.0)", ge=0.5, le=2.0)
    pitch: float = Field(1.0, description="Pitch multiplier (0.5–2.0)", ge=0.5, le=2.0)
    energy: float = Field(
        1.0, description="Energy/volume multiplier (0.5–2.0)", ge=0.5, le=2.0
    )
    style: Optional[str] = Field(None, description="Style preset (happy, sad, calm, …)")
    normalize: bool = Field(True, description="Apply text normalisation")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?",
                "voice": "hi_female",
                "speed": 1.0,
                "pitch": 1.0,
                "energy": 1.0,
                "style": "calm",
                "normalize": True,
            }
        }


class SynthesizeResponse(BaseModel):
    success: bool
    duration: float
    sample_rate: int
    voice: str
    text: str
    inference_time: float


class CloneResponse(BaseModel):
    success: bool
    duration: float
    sample_rate: int
    inference_time: float
    language: str


class VoiceInfo(BaseModel):
    key: str
    name: str
    language_code: str
    gender: str
    loaded: bool
    downloaded: bool
    model_type: str = "vits_jit"


class HealthResponse(BaseModel):
    status: str
    device: str
    loaded_voices: List[str]
    available_voices: int
    style_presets: List[str]


# ---------------------------------------------------------------------------
# Upload validation helper
# ---------------------------------------------------------------------------


def _validate_audio_upload(upload: UploadFile, raw_bytes: bytes) -> None:
    """Raise HTTPException for invalid audio uploads."""
    if upload is None:
        raise HTTPException(status_code=400, detail="speaker_wav is required")

    filename = (upload.filename or "").lower()
    ext = Path(filename).suffix
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=400, detail=f"File too large. Maximum is {mb} MB"
        )

    # Minimal WAV header sanity check
    if ext == ".wav" and len(raw_bytes) < 44:
        raise HTTPException(
            status_code=400, detail="File is too small to be a valid WAV"
        )


# ---------------------------------------------------------------------------
# Helper: audio → WAV bytes
# ---------------------------------------------------------------------------


def _output_to_wav_bytes(output: TTSOutput) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, output.audio, output.sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def _wav_response(
    output: TTSOutput, extra_headers: Optional[Dict[str, str]] = None
) -> Response:
    wav_bytes = _output_to_wav_bytes(output)
    headers = {
        "X-Duration": str(round(output.duration, 4)),
        "X-Sample-Rate": str(output.sample_rate),
        "X-Voice": output.voice,
        "X-Style": output.style or "default",
    }
    if extra_headers:
        headers.update(extra_headers)
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


# ---------------------------------------------------------------------------
# Routes — info / health
# ---------------------------------------------------------------------------


@app.get("/", response_class=JSONResponse, tags=["Info"])
async def root() -> dict:
    """API welcome message."""
    return {
        "message": "VoiceAPI — Multi-lingual TTS (local)",
        "docs": "/docs",
        "health": "/health",
        "synthesize": "/synthesize",
        "clone": "/clone",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check() -> HealthResponse:
    """Return server health and engine status."""
    engine = get_engine()
    return HealthResponse(
        status="healthy",
        device=str(engine.device),
        loaded_voices=engine.get_loaded_voices(),
        available_voices=len(LANGUAGE_CONFIGS),
        style_presets=list(STYLE_PRESETS.keys()),
    )


@app.get("/voices", response_model=List[VoiceInfo], tags=["Info"])
async def list_voices() -> List[VoiceInfo]:
    """List every configured voice with download and load status."""
    engine = get_engine()
    voices = engine.get_available_voices()
    return [
        VoiceInfo(
            key=key,
            name=info["name"],
            language_code=info["code"],
            gender=info["gender"],
            loaded=info["loaded"],
            downloaded=info["downloaded"],
            model_type=info.get("type", "vits_jit"),
        )
        for key, info in voices.items()
    ]


@app.get("/styles", tags=["Info"])
async def list_styles() -> dict:
    """List available style presets and parameter descriptions."""
    return {
        "presets": STYLE_PRESETS,
        "description": {
            "speed": "Speech rate multiplier (0.5–2.0)",
            "pitch": "Pitch multiplier (0.5–2.0)  >1 = higher",
            "energy": "Volume/amplitude multiplier (0.5–2.0)",
        },
    }


@app.get("/languages", tags=["Info"])
async def list_languages() -> dict:
    """Return a mapping of language code → language name."""
    return get_available_languages()


# ---------------------------------------------------------------------------
# Routes — standard synthesis
# ---------------------------------------------------------------------------


@app.post("/synthesize", response_class=Response, tags=["Synthesis"])
async def synthesize_audio(request: SynthesizeRequest) -> Response:
    """
    Synthesize text to speech and return a WAV file.

    Response headers include `X-Duration`, `X-Sample-Rate`, `X-Voice`, `X-Inference-Time`.
    """
    if request.voice not in LANGUAGE_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{request.voice}'. See /voices for options.",
        )

    engine = get_engine()
    try:
        t0 = time.perf_counter()
        output = engine.synthesize(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            pitch=request.pitch,
            energy=request.energy,
            style=request.style,
            normalize_text=request.normalize,
        )
        inference_time = time.perf_counter() - t0
        return _wav_response(
            output, {"X-Inference-Time": str(round(inference_time, 4))}
        )

    except Exception as exc:
        logger.exception("Synthesis error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/synthesize/get", response_class=Response, tags=["Synthesis"])
async def synthesize_get(
    text: str = Query(
        ..., description="Text to synthesize", min_length=1, max_length=1000
    ),
    voice: str = Query("hi_female", description="Voice key"),
    speed: float = Query(1.0, ge=0.5, le=2.0),
    pitch: float = Query(1.0, ge=0.5, le=2.0),
    energy: float = Query(1.0, ge=0.5, le=2.0),
    style: Optional[str] = Query(None),
) -> Response:
    """GET convenience wrapper around /synthesize (handy for quick browser tests)."""
    req = SynthesizeRequest(
        text=text,
        voice=voice,
        speed=speed,
        pitch=pitch,
        energy=energy,
        style=style,
        normalize=True,
    )
    return await synthesize_audio(req)


@app.post("/synthesize/stream", tags=["Synthesis"])
async def synthesize_stream(request: SynthesizeRequest) -> StreamingResponse:
    """Synthesize and stream the WAV audio (chunked transfer)."""
    if request.voice not in LANGUAGE_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown voice '{request.voice}'")

    engine = get_engine()
    try:
        output = engine.synthesize(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            pitch=request.pitch,
            energy=request.energy,
            style=request.style,
            normalize_text=request.normalize,
        )
        buf = io.BytesIO()
        sf.write(buf, output.audio, output.sample_rate, format="WAV")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )
    except Exception as exc:
        logger.exception("Stream synthesis error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Routes — voice cloning (XTTS v2, fully local)
# ---------------------------------------------------------------------------


@app.post("/clone", response_class=Response, tags=["Voice Cloning"])
async def clone_voice(
    text: str = Query(..., description="Text to synthesize in the cloned voice"),
    lang: str = Query(
        "english",
        description=(
            "Language of the output speech. "
            "Supported: english, hindi, bengali, gujarati, marathi, telugu, kannada"
        ),
    ),
    speed: float = Query(1.0, ge=0.5, le=2.0, description="Speed multiplier"),
    pitch: float = Query(1.0, ge=0.5, le=2.0, description="Pitch multiplier"),
    energy: float = Query(1.0, ge=0.5, le=2.0, description="Energy/volume multiplier"),
    style: Optional[str] = Query(None, description="Style preset name"),
    speaker_wav: UploadFile = File(
        ...,
        description=(
            "Reference WAV/MP3 of the speaker whose voice should be cloned. "
            "Best quality with 5–30 seconds of clean speech."
        ),
    ),
) -> Response:
    """
    Clone a voice from a short audio sample using **XTTS v2** (local, no API key).

    - Upload any WAV or MP3 with 5–30 s of the target speaker.
    - XTTS v2 weights (~1.8 GB) are downloaded once and cached in `models/` automatically.
    - Supported languages: english, hindi, bengali, gujarati, marathi, telugu, kannada.
    """
    lang_lower = lang.lower().strip()
    if lang_lower not in XTTS_LANG_MAP:
        supported = ", ".join(sorted(XTTS_LANG_MAP.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported clone language '{lang}'. Supported: {supported}",
        )

    # Read and validate upload
    raw_bytes = await speaker_wav.read()
    _validate_audio_upload(speaker_wav, raw_bytes)

    temp_path: Optional[str] = None
    try:
        # Write to a temp file (XTTS needs a file path)
        suffix = Path(speaker_wav.filename or "ref.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            temp_path = tmp.name

        engine = get_engine()
        t0 = time.perf_counter()
        output = engine.clone_voice(
            text=text,
            speaker_wav_path=temp_path,
            language_code=XTTS_LANG_MAP[lang_lower],
            speed=speed,
            pitch=pitch,
            energy=energy,
            style=style,
            normalize_text=True,
        )
        inference_time = time.perf_counter() - t0

        return _wav_response(
            output,
            {
                "X-Language": lang_lower,
                "X-Inference-Time": str(round(inference_time, 4)),
                "Content-Disposition": "attachment; filename=cloned_output.wav",
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Clone error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Routes — hackathon /Get_Inference endpoint
# ---------------------------------------------------------------------------


@app.api_route("/Get_Inference", methods=["GET", "POST"], tags=["Hackathon"])
async def get_inference(
    text: str = Query(
        ...,
        description="Input text. For English, must be lowercase.",
    ),
    lang: str = Query(
        ...,
        description=(
            "Language name. One of: bhojpuri, bengali, english, gujarati, hindi, "
            "chhattisgarhi, kannada, magahi, maithili, marathi, telugu"
        ),
    ),
    speaker_wav: UploadFile = File(
        ...,
        description="Reference WAV (required by spec; used for voice cloning when lang supports it).",
    ),
) -> StreamingResponse:
    """
    Hackathon specification endpoint.

    Accepts `text`, `lang`, and `speaker_wav` and returns synthesised WAV audio.

    - For languages supported by XTTS (english, hindi, bengali, gujarati, marathi,
      telugu, kannada): performs **voice cloning** using the uploaded WAV.
    - For other languages (bhojpuri, chhattisgarhi, maithili, magahi): uses the
      pre-trained VITS/Coqui model for that language (speaker_wav accepted but not used).
    """
    engine = get_engine()
    lang_lower = lang.lower().strip()

    if lang_lower not in LANG_TO_VOICE:
        supported = ", ".join(sorted(LANG_TO_VOICE.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{lang}'. Supported: {supported}",
        )

    # Enforce lowercase English (per spec)
    if lang_lower == "english":
        text = text.lower()

    # Read and validate the uploaded audio
    raw_bytes = await speaker_wav.read()
    _validate_audio_upload(speaker_wav, raw_bytes)

    logger.info(
        "Get_Inference: lang=%s, text_len=%d, wav_bytes=%d",
        lang_lower,
        len(text),
        len(raw_bytes),
    )

    # ── Voice cloning path (XTTS-supported languages) ────────────────────
    if lang_lower in XTTS_LANG_MAP:
        temp_path: Optional[str] = None
        try:
            suffix = Path(speaker_wav.filename or "ref.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw_bytes)
                temp_path = tmp.name

            output = engine.clone_voice(
                text=text,
                speaker_wav_path=temp_path,
                language_code=XTTS_LANG_MAP[lang_lower],
                normalize_text=True,
            )
        except Exception as exc:
            logger.warning(
                "Clone failed for %s (%s), falling back to standard VITS",
                lang_lower,
                exc,
            )
            # Fallback to standard synthesis
            voice = LANG_TO_VOICE[lang_lower]
            output = engine.synthesize(text=text, voice=voice, normalize_text=True)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    # ── Standard synthesis path (Bhojpuri, Chhattisgarhi, Maithili, Magahi) ──
    else:
        voice = LANG_TO_VOICE[lang_lower]
        try:
            output = engine.synthesize(text=text, voice=voice, normalize_text=True)
        except Exception as exc:
            logger.exception("Synthesis error for lang=%s", lang_lower)
            raise HTTPException(status_code=500, detail=str(exc))

    buf = io.BytesIO()
    sf.write(buf, output.audio, output.sample_rate, format="WAV")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=output.wav",
            "X-Duration": str(round(output.duration, 4)),
            "X-Sample-Rate": str(output.sample_rate),
            "X-Language": lang_lower,
            "X-Voice": output.voice,
        },
    )


# ---------------------------------------------------------------------------
# Routes — model management
# ---------------------------------------------------------------------------


@app.post("/preload", tags=["Model Management"])
async def preload_voice(voice: str) -> dict:
    """Load a voice model into memory so the first synthesis request is faster."""
    if voice not in LANGUAGE_CONFIGS:
        raise HTTPException(
            status_code=400, detail=f"Unknown voice '{voice}'. See /voices."
        )
    engine = get_engine()
    try:
        engine.load_voice(voice)
        return {"message": f"Voice '{voice}' loaded successfully"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/unload", tags=["Model Management"])
async def unload_voice(voice: str) -> dict:
    """Unload a voice model from memory to free RAM."""
    engine = get_engine()
    engine.unload_voice(voice)
    return {"message": f"Voice '{voice}' unloaded"}


# ---------------------------------------------------------------------------
# Routes — batch synthesis
# ---------------------------------------------------------------------------


@app.post("/batch", tags=["Synthesis"])
async def batch_synthesize(
    texts: List[str],
    voice: str = "hi_female",
    speed: float = 1.0,
) -> List[dict]:
    """
    Synthesise a list of texts with the same voice.

    Returns a list of objects with `text`, `duration`, and `audio_base64` (WAV).
    """
    import base64

    if voice not in LANGUAGE_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown voice '{voice}'")

    engine = get_engine()
    results = []
    for t in texts:
        try:
            output = engine.synthesize(t, voice, speed)
            wav = _output_to_wav_bytes(output)
            results.append(
                {
                    "text": t,
                    "audio_base64": base64.b64encode(wav).decode(),
                    "duration": round(output.duration, 4),
                    "sample_rate": output.sample_rate,
                }
            )
        except Exception as exc:
            results.append({"text": t, "error": str(exc)})

    return results


# ---------------------------------------------------------------------------
# Startup / shutdown events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("VoiceAPI starting — preloading all synthesis models…")
    try:
        engine = get_engine()
        results = engine.preload_all_voices()
        n_ok = sum(1 for v in results.values() if v)
        n_fail = sum(1 for v in results.values() if not v)
        logger.info(
            "Preload complete: %d/%d voices loaded (%d failed)",
            n_ok,
            len(results),
            n_fail,
        )
    except Exception as exc:
        logger.error("Preload encountered an error — server will still start: %s", exc)
    logger.info("Docs: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("VoiceAPI local server shutting down.")


# ---------------------------------------------------------------------------
# Programmatic entry point
# ---------------------------------------------------------------------------


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info",
) -> None:
    """Start the API server (called by `python -m src.cli serve`)."""
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
