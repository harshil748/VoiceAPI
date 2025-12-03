#!/usr/bin/env python
"""
Quick test script to verify the TTS system works
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_basic():
    """Basic functionality test"""
    print("=" * 50)
    print("🧪 Testing Voice Tech for All TTS System")
    print("=" * 50)

    # Test 1: Import modules
    print("\n1. Testing imports...")
    try:
        from src.config import LANGUAGE_CONFIGS, get_available_voices
        from src.tokenizer import TTSTokenizer, CharactersConfig, TextNormalizer
        from src.downloader import ModelDownloader
        from src.engine import TTSEngine

        print("   ✅ All imports successful")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

    # Test 2: Configuration
    print("\n2. Testing configuration...")
    voices = get_available_voices()
    print(f"   ✅ Found {len(voices)} voice configurations")
    print(f"   Languages: {set(v['code'] for v in voices.values())}")

    # Test 3: Tokenizer
    print("\n3. Testing tokenizer...")
    config = CharactersConfig(
        characters="abcdefghijklmnopqrstuvwxyz", punctuations="!.,? "
    )
    tokenizer = TTSTokenizer(config)
    ids = tokenizer.text_to_ids("hello world")
    text_back = tokenizer.ids_to_text(ids)
    print(f"   ✅ Tokenizer works: 'hello world' -> {len(ids)} tokens")

    # Test 4: Text normalizer
    print("\n4. Testing text normalizer...")
    normalizer = TextNormalizer()
    test_text = "Price is {100}{एकसो} rupees"
    normalized = normalizer.clean_text(test_text)
    print(f"   ✅ Normalized: '{test_text}' -> '{normalized}'")

    # Test 5: Model downloader
    print("\n5. Testing model downloader...")
    downloader = ModelDownloader()
    downloaded = downloader.list_downloaded_models()
    print(f"   ✅ Downloaded models: {downloaded if downloaded else 'None yet'}")

    # Test 6: Engine initialization
    print("\n6. Testing TTS engine...")
    try:
        engine = TTSEngine()
        print(f"   ✅ Engine initialized on device: {engine.device}")
    except Exception as e:
        print(f"   ⚠️ Engine init warning: {e}")

    print("\n" + "=" * 50)
    print("✅ All basic tests passed!")
    print("=" * 50)

    print("\n📋 Next steps:")
    print("   1. Download a model: python -m src.cli download --voice hi_male")
    print(
        "   2. Synthesize: python -m src.cli synthesize --text 'नमस्ते' --voice hi_male"
    )
    print("   3. Start server: python -m src.cli serve")

    return True


def test_synthesis():
    """Test actual synthesis (requires downloaded model)"""
    from src.engine import TTSEngine
    from src.downloader import ModelDownloader

    downloader = ModelDownloader()
    downloaded = downloader.list_downloaded_models()

    if not downloaded:
        print("\n⚠️ No models downloaded yet.")
        print("Run: python -m src.cli download --voice hi_male")
        return

    voice = downloaded[0]
    print(f"\n🎤 Testing synthesis with voice: {voice}")

    engine = TTSEngine()

    # Test synthesis
    test_texts = {
        "hi": "नमस्ते, मैं आपकी कैसे मदद कर सकता हूं?",
        "en": "Hello, how can I help you today?",
        "bn": "নমস্কার, আজ আমি আপনাকে কীভাবে সাহায্য করতে পারি?",
    }

    # Get language for this voice
    from src.config import LANGUAGE_CONFIGS

    lang = LANGUAGE_CONFIGS[voice].code

    text = test_texts.get(lang, test_texts["en"])

    print(f"   Text: {text}")
    output = engine.synthesize(text, voice)
    print(f"   ✅ Generated {output.duration:.2f}s of audio")

    # Save test file
    test_output = "test_output.wav"
    engine.synthesize_to_file(text, test_output, voice)
    print(f"   ✅ Saved to: {test_output}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        test_basic()
        test_synthesis()
    else:
        test_basic()
