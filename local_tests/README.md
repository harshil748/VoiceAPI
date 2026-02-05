# Local Tests Output Directory

This directory contains outputs from local API testing.

## Structure

- `reference.wav` - Auto-generated reference audio for testing
- `{language}_{timestamp}.wav` - Test outputs for each language

## How to Run Tests

### Step 1: Start the Local API Server

```bash
# In Terminal 1
cd /Users/harshilpatel/Developer/hackathons/TTS
source tts/bin/activate
python -m src.cli serve --port 8000
```

### Step 2: Run the Test Script

```bash
# In Terminal 2 (new tab)
cd /Users/harshilpatel/Developer/hackathons/TTS
source tts/bin/activate
python local_tests/test_local_api.py
```

### Step 3: Check Outputs

All generated WAV files will be saved in `local_tests/outputs/`

## Test Coverage

- ✅ Hindi
- ✅ Bengali
- ✅ English
- ✅ Gujarati
- ✅ Kannada
- ✅ Marathi
- ✅ Telugu

## Expected Output

```
============================================================
LOCAL API TEST SUITE
============================================================
API Endpoint: http://localhost:8000/Get_Inference
Output Directory: local_tests/outputs
Total Tests: 7

✓ Generated reference WAV: local_tests/outputs/reference.wav

============================================================
Testing: HINDI
Text: नमस्ते, आप कैसे हैं?
============================================================
✅ SUCCESS
   Status: 200
   Latency: 0.85s
   Output: hindi_20260205_143022.wav
   Size: 245,760 bytes

[... similar for other languages ...]

============================================================
TEST SUMMARY
============================================================
Total: 7
Passed: 7
Failed: 0
Avg Latency: 0.72s

Outputs saved to: local_tests/outputs
============================================================
```

## Troubleshooting

### Connection Error

```
❌ CONNECTION ERROR
   Make sure the API server is running on localhost:8000
```

**Solution**: Start the server with `python -m src.cli serve --port 8000`

### 422 Unprocessable Entity

**Solution**: Check that all parameters (text, lang, speaker_wav) are correct

### Models Not Found

**Solution**: Download models first with `python -m src.cli download --lang hi`
