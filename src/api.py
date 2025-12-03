"""
REST API Server for Multi-lingual TTS
FastAPI-based server with OpenAPI documentation
"""

import os
import io
import time
import logging
import tempfile
from typing import Optional, List
from pathlib import Path
import numpy as np

from fastapi import FastAPI, HTTPException, Query, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
import soundfile as sf

from .engine import TTSEngine, TTSOutput
from .config import (
    LANGUAGE_CONFIGS,
    get_available_languages,
    get_available_voices,
    STYLE_PRESETS,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voice Tech for All - Multi-lingual TTS API",
    description="""
    A multi-lingual Text-to-Speech API supporting 10+ Indian languages.
    
    ## Features
    - 10 Indian languages with male/female voices
    - Real-time speech synthesis
    - Text normalization for Indian languages
    - Speed control
    - Multiple audio formats (WAV, MP3)
    
    ## Supported Languages
    Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, 
    Chhattisgarhi, Maithili, Magahi, English
    
    ## Use Case
    Built for an LLM-based healthcare assistant for pregnant mothers
    in low-income communities.
    """,
    version="1.0.0",
    contact={
        "name": "Voice Tech for All Hackathon",
        "url": "https://huggingface.co/SYSPIN",
    },
    license_info={
        "name": "CC BY 4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize TTS Engine (lazy loading)
_engine: Optional[TTSEngine] = None


def get_engine() -> TTSEngine:
    """Get or create TTS engine instance"""
    global _engine
    if _engine is None:
        _engine = TTSEngine(device="auto")
    return _engine


# Request/Response Models
class SynthesizeRequest(BaseModel):
    """Request body for text synthesis"""

    text: str = Field(
        ..., description="Text to synthesize", min_length=1, max_length=5000
    )
    voice: str = Field(
        "hi_male", description="Voice key (e.g., hi_male, bn_female, gu_mms)"
    )
    speed: float = Field(1.0, description="Speech speed (0.5-2.0)", ge=0.5, le=2.0)
    pitch: float = Field(1.0, description="Pitch multiplier (0.5-2.0)", ge=0.5, le=2.0)
    energy: float = Field(1.0, description="Energy/volume (0.5-2.0)", ge=0.5, le=2.0)
    style: Optional[str] = Field(
        None, description="Style preset (happy, sad, calm, excited, etc.)"
    )
    normalize: bool = Field(True, description="Apply text normalization")

    class Config:
        schema_extra = {
            "example": {
                "text": "નમસ્તે, હું તમારી કેવી રીતે મદદ કરી શકું?",
                "voice": "gu_mms",
                "speed": 1.0,
                "pitch": 1.0,
                "energy": 1.0,
                "style": "calm",
                "normalize": True,
            }
        }


class SynthesizeResponse(BaseModel):
    """Response metadata for synthesis"""

    success: bool
    duration: float
    sample_rate: int
    voice: str
    text: str
    inference_time: float


class VoiceInfo(BaseModel):
    """Information about a voice"""

    key: str
    name: str
    language_code: str
    gender: str
    loaded: bool
    downloaded: bool
    model_type: str = "vits"


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    device: str
    loaded_voices: List[str]
    available_voices: int
    style_presets: List[str]


# API Endpoints
@app.get("/", response_class=JSONResponse)
async def root():
    """API root - welcome message"""
    return {
        "message": "Voice Tech for All - Multi-lingual TTS API",
        "docs": "/docs",
        "health": "/health",
        "synthesize": "/synthesize",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    engine = get_engine()
    return HealthResponse(
        status="healthy",
        device=str(engine.device),
        loaded_voices=engine.get_loaded_voices(),
        available_voices=len(LANGUAGE_CONFIGS),
        style_presets=list(STYLE_PRESETS.keys()),
    )


@app.get("/voices", response_model=List[VoiceInfo])
async def list_voices():
    """List all available voices"""
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
            model_type=info.get("type", "vits"),
        )
        for key, info in voices.items()
    ]


@app.get("/styles")
async def list_styles():
    """List available style presets for prosody control"""
    return {
        "presets": STYLE_PRESETS,
        "description": {
            "speed": "Speech rate multiplier (0.5-2.0)",
            "pitch": "Pitch multiplier (0.5-2.0), >1 = higher",
            "energy": "Volume/energy multiplier (0.5-2.0)",
        },
    }


@app.get("/languages")
async def list_languages():
    """List supported languages"""
    return get_available_languages()


@app.post("/synthesize", response_class=Response)
async def synthesize_audio(request: SynthesizeRequest):
    """
    Synthesize speech from text

    Returns WAV audio file directly
    """
    engine = get_engine()

    # Validate voice
    if request.voice not in LANGUAGE_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {request.voice}. Use /voices to see available options.",
        )

    try:
        start_time = time.time()

        # Synthesize
        output = engine.synthesize(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            pitch=request.pitch,
            energy=request.energy,
            style=request.style,
            normalize_text=request.normalize,
        )

        inference_time = time.time() - start_time

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, output.audio, output.sample_rate, format="WAV")
        buffer.seek(0)

        # Return audio with metadata headers
        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "X-Duration": str(output.duration),
                "X-Sample-Rate": str(output.sample_rate),
                "X-Voice": output.voice,
                "X-Style": output.style or "default",
                "X-Inference-Time": str(inference_time),
            },
        )

    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/stream")
async def synthesize_stream(request: SynthesizeRequest):
    """
    Synthesize speech and stream the audio

    Returns streaming WAV audio
    """
    engine = get_engine()

    if request.voice not in LANGUAGE_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {request.voice}")

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

        # Create streaming response
        buffer = io.BytesIO()
        sf.write(buffer, output.audio, output.sample_rate, format="WAV")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/synthesize/get")
async def synthesize_get(
    text: str = Query(
        ..., description="Text to synthesize", min_length=1, max_length=1000
    ),
    voice: str = Query("hi_male", description="Voice key"),
    speed: float = Query(1.0, description="Speech speed", ge=0.5, le=2.0),
    pitch: float = Query(1.0, description="Pitch", ge=0.5, le=2.0),
    energy: float = Query(1.0, description="Energy", ge=0.5, le=2.0),
    style: Optional[str] = Query(None, description="Style preset"),
):
    """
    GET endpoint for simple synthesis

    Useful for testing and simple integrations
    """
    request = SynthesizeRequest(
        text=text, voice=voice, speed=speed, pitch=pitch, energy=energy, style=style
    )
    return await synthesize_audio(request)


@app.post("/preload")
async def preload_voice(voice: str):
    """Preload a voice model into memory"""
    engine = get_engine()

    if voice not in LANGUAGE_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {voice}")

    try:
        engine.load_voice(voice)
        return {"message": f"Voice {voice} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload_voice(voice: str):
    """Unload a voice model from memory"""
    engine = get_engine()
    engine.unload_voice(voice)
    return {"message": f"Voice {voice} unloaded"}


@app.post("/batch")
async def batch_synthesize(
    texts: List[str], voice: str = "hi_male", speed: float = 1.0
):
    """
    Synthesize multiple texts

    Returns a list of base64-encoded audio
    """
    import base64

    engine = get_engine()

    if voice not in LANGUAGE_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {voice}")

    results = []
    for text in texts:
        output = engine.synthesize(text, voice, speed)

        buffer = io.BytesIO()
        sf.write(buffer, output.audio, output.sample_rate, format="WAV")
        buffer.seek(0)

        results.append(
            {
                "text": text,
                "audio_base64": base64.b64encode(buffer.read()).decode(),
                "duration": output.duration,
            }
        )

    return results


# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting TTS API server...")
    # Optionally preload default voice
    # get_engine().load_voice("hi_male")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down TTS API server...")


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server"""
    import uvicorn

    uvicorn.run("src.api:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    start_server()
