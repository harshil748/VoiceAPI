import os
import wave
import numpy as np
import requests

# This script mirrors the hackathon spec Python snippet for GET /Get_Inference
# with a generated reference WAV so it runs end-to-end.

base_url = "https://harshil748-voiceapi.hf.space/Get_Inference"
WavPath = "tests/reference.wav"

params = {
    "text": "ಮಾದರಿಯು ಸರಿಯಾಗಿ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತಿದೆಯೇ ಎಂದು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಲು ಬಳಸಲಾಗುವ ಪರೀಕ್ಷಾ ವಾಕ್ಯ ಇದು.",
    "lang": "kannada",
}


def ensure_reference_wav(path: str, sample_rate: int = 22050, duration: float = 1.0):
    """Generate a simple reference WAV (sine tone) if none exists."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 220 * t)
    pcm = np.int16(tone * 32767)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def main():
    ensure_reference_wav(WavPath)
    with open(WavPath, "rb") as AudioFile:
        response = requests.get(
            base_url, params=params, files={"speaker_wav": AudioFile}, timeout=120
        )
    if response.status_code == 200:
        output_path = "tests/output.wav"
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Success: saved {output_path}")
    else:
        print(f"Request failed: {response.status_code}\n{response.text}")


if __name__ == "__main__":
    main()
