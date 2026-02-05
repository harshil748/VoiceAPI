import os
import wave
import time
import numpy as np
import requests
from datetime import datetime
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000/Get_Inference"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
REFERENCE_WAV = OUTPUTS_DIR / "reference.wav"

# Test cases: language + sample text
TEST_CASES = [
    {"lang": "hindi", "text": "नमस्ते, आप कैसे हैं?"},
    {"lang": "bengali", "text": "আপনি কেমন আছেন?"},
    {"lang": "english", "text": "hello how are you"},
    {"lang": "gujarati", "text": "તમે કેમ છો?"},
    {"lang": "kannada", "text": "ನೀವು ಹೇಗಿದ್ದೀರಿ?"},
    {"lang": "marathi", "text": "तुम्ही कसे आहात?"},
    {"lang": "telugu", "text": "మీరు ఎలా ఉన్నారు?"},
]


def generate_reference_wav(sample_rate: int = 22050, duration: float = 1.0):
    """Generate a simple sine wave WAV file as reference"""
    if REFERENCE_WAV.exists():
        print(f"✓ Reference WAV already exists: {REFERENCE_WAV}")
        return

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 220 * t)  # 220 Hz tone
    pcm = np.int16(tone * 32767)

    with wave.open(str(REFERENCE_WAV), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    print(f"Generated reference WAV: {REFERENCE_WAV}")


def test_api_endpoint(lang: str, text: str):
    """Test API with given language and text"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{lang}_{timestamp}.wav"
    output_path = OUTPUTS_DIR / output_filename

    print(f"\n{'='*60}")
    print(f"Testing: {lang.upper()}")
    print(f"Text: {text}")
    print(f"{'='*60}")

    params = {
        "text": text,
        "lang": lang,
    }

    try:
        start_time = time.time()

        with open(REFERENCE_WAV, "rb") as audio_file:
            response = requests.get(
                BASE_URL, params=params, files={"speaker_wav": audio_file}, timeout=60
            )

        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)

            file_size = output_path.stat().st_size
            print(f"SUCCESS")
            print(f"Status: {response.status_code}")
            print(f"Latency: {elapsed_time:.2f}s")
            print(f"Output: {output_filename}")
            print(f"Size: {file_size:,} bytes")

            return {
                "lang": lang,
                "status": "success",
                "latency": elapsed_time,
                "size": file_size,
                "output": output_filename,
            }
        else:
            print(f"FAILED")
            print(f"Status: {response.status_code}")
            print(f"Error: {response.text[:200]}")

            return {"lang": lang, "status": "failed", "error": response.status_code}

    except requests.exceptions.ConnectionError:
        print(f"CONNECTION ERROR")
        print(f"Make sure the API server is running on localhost:8000")
        return {"lang": lang, "status": "connection_error"}

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"lang": lang, "status": "error", "error": str(e)}


def run_all_tests():
    """Run all test cases"""
    print("\n" + "=" * 60)
    print("LOCAL API TEST SUITE")
    print("=" * 60)
    print(f"API Endpoint: {BASE_URL}")
    print(f"Output Directory: {OUTPUTS_DIR}")
    print(f"Total Tests: {len(TEST_CASES)}")

    # Generate reference WAV
    generate_reference_wav()

    # Run all tests
    results = []
    for test_case in TEST_CASES:
        result = test_api_endpoint(test_case["lang"], test_case["text"])
        results.append(result)
        time.sleep(0.5)  # Small delay between requests

    # Summary
    print("TEST SUMMARY")
 

    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = len(results) - success_count

    print(f"Total: {len(results)}")
    print(f"Passed: {success_count}")
    print(f"Failed: {failed_count}")

    if success_count > 0:
        avg_latency = (
            sum(r.get("latency", 0) for r in results if r.get("status") == "success")
            / success_count
        )
        print(f"Avg Latency: {avg_latency:.2f}s")

    print(f"\nOutputs saved to: {OUTPUTS_DIR}")
    

    # List all output files
    output_files = sorted(OUTPUTS_DIR.glob("*.wav"))
    if output_files:
        print("\nGenerated Files:")
        for f in output_files:
            if f.name != "reference.wav":
                size_kb = f.stat().st_size / 1024
                print(f"  • {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    run_all_tests()
