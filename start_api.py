#!/usr/bin/env python3
"""
VoiceAPI — Local API Server Entry Point
========================================
Run the FastAPI backend without the web UI.

Usage:
    python start_api.py                        # default: 0.0.0.0:8000
    python start_api.py --port 8001
    python start_api.py --host 127.0.0.1 --port 8000 --reload
    python start_api.py --log-level debug

Or via uvicorn directly:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `src` is importable when this
# script is run from any working directory (e.g. `python TTS/start_api.py`).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change cwd to project root so relative paths (models/, etc.) resolve correctly
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voiceapi.start")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Start the VoiceAPI local FastAPI server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Network interface to bind (use 127.0.0.1 to restrict to localhost)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="TCP port for the API server",
    )
    p.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot-reload (development only — slower startup)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (>1 disables --reload)",
    )
    p.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level",
    )
    p.add_argument(
        "--preload",
        nargs="*",
        metavar="VOICE_KEY",
        help=(
            "Voice keys to load into memory at startup, e.g. "
            "--preload hi_female en_female"
        ),
    )
    return p.parse_args()


def check_dependencies() -> None:
    """Warn early about missing required packages."""
    missing = []
    for pkg, import_name in [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("torch", "torch"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("transformers", "transformers"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error(
            "Missing required packages: %s\n"
            "Install them with:  pip install -r requirements.txt",
            ", ".join(missing),
        )
        sys.exit(1)


def check_models_dir() -> None:
    """Print a warning if the models directory looks empty."""
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        logger.warning(
            "models/ directory not found. Run:  python -m src.cli download --all"
        )
        return

    pt_files = list(models_dir.rglob("*.pt")) + list(models_dir.rglob("*.pth"))
    if not pt_files:
        logger.warning(
            "No .pt / .pth files found in models/. "
            "Run:  python -m src.cli download --all"
        )
    else:
        logger.info("Found %d model file(s) in models/", len(pt_files))


def main() -> None:
    args = parse_args()

    logger.info("=" * 56)
    logger.info("  VoiceAPI — Local Server")
    logger.info("  http://%s:%d", args.host, args.port)
    logger.info("  Docs: http://localhost:%d/docs", args.port)
    logger.info("=" * 56)

    check_dependencies()
    check_models_dir()

    # If specific voices should be preloaded, set an env var that the
    # engine can pick up, or pass them directly after import.
    if args.preload:
        os.environ["PRELOAD_VOICES"] = ",".join(args.preload)
        logger.info("Will preload voices: %s", ", ".join(args.preload))

    import uvicorn

    # When workers > 1, reload must be disabled (uvicorn restriction).
    reload = args.reload and args.workers == 1

    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=reload,
        workers=args.workers if not reload else 1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
