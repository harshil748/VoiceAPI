#!/usr/bin/env python
"""
CLI for Voice Tech for All TTS System
"""
import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Voice Tech for All - Multi-lingual TTS System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Hindi models
  python -m src.cli download --lang hi
  
  # Download all models
  python -m src.cli download --all
  
  # Synthesize text
  python -m src.cli synthesize --text "नमस्ते" --voice hi_male --output hello.wav
  
  # Start API server
  python -m src.cli serve --port 8000
  
  # List available voices
  python -m src.cli list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download TTS models")
    download_parser.add_argument("--voice", type=str, help="Specific voice to download")
    download_parser.add_argument(
        "--lang", type=str, help="Download all voices for a language"
    )
    download_parser.add_argument(
        "--all", action="store_true", help="Download all models"
    )
    download_parser.add_argument(
        "--force", action="store_true", help="Force re-download"
    )

    # Synthesize command
    synth_parser = subparsers.add_parser("synthesize", help="Synthesize text to speech")
    synth_parser.add_argument(
        "--text", "-t", type=str, required=True, help="Text to synthesize"
    )
    synth_parser.add_argument(
        "--voice", "-v", type=str, default="hi_male", help="Voice to use"
    )
    synth_parser.add_argument(
        "--output", "-o", type=str, default="output.wav", help="Output file"
    )
    synth_parser.add_argument(
        "--speed", "-s", type=float, default=1.0, help="Speech speed"
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind"
    )
    serve_parser.add_argument(
        "--port", "-p", type=int, default=8000, help="Port to bind"
    )
    serve_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available voices")

    args = parser.parse_args()

    if args.command == "download":
        from src.downloader import ModelDownloader

        downloader = ModelDownloader()

        if args.voice:
            downloader.download_model(args.voice, force=args.force)
        elif args.lang:
            downloader.download_language(args.lang, force=args.force)
        elif args.all:
            downloader.download_all_models(force=args.force)
        else:
            download_parser.print_help()

    elif args.command == "synthesize":
        from src.engine import TTSEngine

        engine = TTSEngine()

        print(f"Synthesizing: {args.text}")
        print(f"Voice: {args.voice}")

        output_path = engine.synthesize_to_file(
            text=args.text, output_path=args.output, voice=args.voice, speed=args.speed
        )
        print(f"Saved to: {output_path}")

    elif args.command == "serve":
        from src.api import start_server

        print(f"Starting server on {args.host}:{args.port}")
        start_server(host=args.host, port=args.port, reload=args.reload)

    elif args.command == "list":
        from src.config import LANGUAGE_CONFIGS
        from src.downloader import ModelDownloader

        downloader = ModelDownloader()

        print("\n📢 Available TTS Voices:\n")
        print(f"{'Voice Key':<15} {'Language':<15} {'Gender':<10} {'Downloaded':<12}")
        print("-" * 55)

        for key, config in LANGUAGE_CONFIGS.items():
            downloaded = "✓" if downloader.get_model_path(key) else "✗"
            gender = "Male" if "male" in key else "Female"
            print(f"{key:<15} {config.name:<15} {gender:<10} {downloaded:<12}")

        print(f"\nTotal: {len(LANGUAGE_CONFIGS)} voices")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
