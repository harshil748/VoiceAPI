#!/usr/bin/env python3
"""
Download open-source reference audio from web and test live clone endpoint
language-by-language against the deployed API.

Outputs:
- local_tests/references/open_source_reference.wav
- local_tests/outputs/live_clone_tests/<timestamp>/*.wav
- local_tests/outputs/live_clone_tests/<timestamp>/report.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests

API_BASE = "https://harshil748-voiceapi.hf.space"

LANGUAGES = [
    "hindi",
    "bengali",
    "english",
    "gujarati",
    "marathi",
    "telugu",
    "kannada",
    "bhojpuri",
    "chhattisgarhi",
    "maithili",
    "magahi",
]

TEST_TEXT: Dict[str, str] = {
    "hindi": "नमस्ते, यह वॉइस क्लोन परीक्षण है।",
    "bengali": "নমস্কার, এটি একটি ভয়েস ক্লোন পরীক্ষা।",
    "english": "hello, this is a voice clone test.",
    "gujarati": "નમસ્તે, આ વોઇસ ક્લોન ટેસ્ટ છે.",
    "marathi": "नमस्कार, ही व्हॉइस क्लोन चाचणी आहे.",
    "telugu": "నమస్కారం, ఇది వాయిస్ క్లోన్ పరీక్ష.",
    "kannada": "ನಮಸ್ಕಾರ, ಇದು ವಾಯ್ಸ್ ಕ್ಲೋನ್ ಪರೀಕ್ಷೆ.",
    "bhojpuri": "प्रणाम, ई आवाज क्लोन टेस्ट बा।",
    "chhattisgarhi": "नमस्कार, ये आवाज क्लोन जांच हे।",
    "maithili": "प्रणाम, ई आवाज क्लोन परीक्षण अछि।",
    "magahi": "प्रणाम, ई आवाज क्लोन टेस्ट हई।",
}

# Open-source WAV sample (MIT-licensed repo: free-spoken-digit-dataset)
OPEN_SOURCE_WAV_URL = (
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/"
    "recordings/0_george_0.wav"
)

ROOT = Path(__file__).resolve().parent
REF_DIR = ROOT / "references"
OUT_BASE = ROOT / "outputs" / "live_clone_tests"


def ensure_reference_file() -> Path:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = REF_DIR / "open_source_reference.wav"

    if ref_path.exists() and ref_path.stat().st_size > 1000:
        return ref_path

    print(f"Downloading open-source reference audio from: {OPEN_SOURCE_WAV_URL}")
    r = requests.get(OPEN_SOURCE_WAV_URL, timeout=30)
    r.raise_for_status()

    with open(ref_path, "wb") as f:
        f.write(r.content)

    if ref_path.stat().st_size < 1000:
        raise RuntimeError("Downloaded reference WAV seems too small")

    return ref_path


def run_clone_tests(reference_wav: Path) -> Dict[str, object]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []

    for lang in LANGUAGES:
        text = TEST_TEXT[lang]
        print(f"Testing language: {lang}")

        url = (
            f"{API_BASE}/clone"
            f"?text={requests.utils.quote(text)}"
            f"&lang={lang}&style=default&speed=1&pitch=1&energy=1"
        )

        row: Dict[str, object] = {
            "language": lang,
            "endpoint": "/clone",
            "status_code": None,
            "ok": False,
            "output_file": None,
            "error": None,
        }

        try:
            with open(reference_wav, "rb") as f:
                response = requests.post(
                    url,
                    files={"speaker_wav": f},
                    timeout=120,
                )

            row["status_code"] = response.status_code

            if response.status_code == 200:
                out_file = out_dir / f"clone_{lang}.wav"
                with open(out_file, "wb") as f:
                    f.write(response.content)
                row["ok"] = True
                row["output_file"] = str(out_file.relative_to(ROOT.parent))
            else:
                row["error"] = response.text[:500]

        except Exception as exc:
            row["error"] = str(exc)

        results.append(row)

    passed = sum(1 for r in results if r["ok"])

    report = {
        "api_base": API_BASE,
        "reference_wav": str(reference_wav.relative_to(ROOT.parent)),
        "total_languages_tested": len(LANGUAGES),
        "passed": passed,
        "failed": len(LANGUAGES) - passed,
        "results": results,
    }

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return {
        "out_dir": out_dir,
        "report_path": report_path,
        "report": report,
    }


def main() -> None:
    ref_wav = ensure_reference_file()
    result = run_clone_tests(ref_wav)

    report = result["report"]
    print("\n=== SUMMARY ===")
    print(f"API: {report['api_base']}")
    print(f"Reference: {report['reference_wav']}")
    print(f"Total: {report['total_languages_tested']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    print(f"Report: {result['report_path']}")
    print(f"Outputs: {result['out_dir']}")


if __name__ == "__main__":
    main()
