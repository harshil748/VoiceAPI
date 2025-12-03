"""
TTS Tokenizer for VITS models
Adapted from Coqui TTS for SYSPIN models
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CharactersConfig:
    """Character configuration for tokenizer"""

    characters: str = ""
    punctuations: str = "!¡'(),-.:;¿? "
    pad: str = "<PAD>"
    eos: str = "<EOS>"
    bos: str = "<BOS>"
    blank: str = "<BLNK>"
    phonemes: Optional[str] = None


class TTSTokenizer:
    """Tokenizer for TTS models"""

    def __init__(self, config: CharactersConfig):
        self.config = config
        self.pad = config.pad
        self.eos = config.eos
        self.bos = config.bos
        self.blank = config.blank
        self.characters = config.characters
        self.punctuations = config.punctuations

        # Build character to ID mapping
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary from characters"""
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        # Special tokens
        idx = 0
        for special in [self.pad, self.eos, self.bos, self.blank]:
            if special:
                self.char_to_id[special] = idx
                self.id_to_char[idx] = special
                idx += 1

        # Characters
        for char in self.characters:
            if char not in self.char_to_id:
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char
                idx += 1

        # Punctuations
        for char in self.punctuations:
            if char not in self.char_to_id:
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char
                idx += 1

        self.vocab_size = len(self.char_to_id)

    def text_to_ids(self, text: str, add_blank: bool = True) -> List[int]:
        """Convert text to token IDs"""
        text = self._clean_text(text)
        ids = []

        for char in text:
            if char in self.char_to_id:
                if add_blank and self.blank and self.blank in self.char_to_id:
                    ids.append(self.char_to_id[self.blank])
                ids.append(self.char_to_id[char])

        # Add final blank
        if add_blank and self.blank and self.blank in self.char_to_id:
            ids.append(self.char_to_id[self.blank])

        return ids

    def ids_to_text(self, ids: List[int]) -> str:
        """Convert token IDs back to text"""
        chars = []
        for id in ids:
            if id in self.id_to_char:
                char = self.id_to_char[id]
                if char not in [self.pad, self.eos, self.bos, self.blank]:
                    chars.append(char)
        return "".join(chars)

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase for some languages
        # text = text.lower()  # Keep original case for Indic scripts

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @classmethod
    def from_chars_file(cls, chars_file: str) -> "TTSTokenizer":
        """Create tokenizer from chars.txt file"""
        with open(chars_file, "r", encoding="utf-8") as f:
            chars = f.read().strip("\n")

        config = CharactersConfig(
            characters=chars,
            punctuations="!¡'(),-.:;¿? ",
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
        )

        return cls(config)


# Text normalization for Indian languages
class TextNormalizer:
    """Text normalizer for Indian languages"""

    # Number words in Hindi
    HINDI_NUMBERS = {
        "0": "शून्य",
        "1": "एक",
        "2": "दो",
        "3": "तीन",
        "4": "चार",
        "5": "पाँच",
        "6": "छह",
        "7": "सात",
        "8": "आठ",
        "9": "नौ",
        "10": "दस",
        "100": "सौ",
        "1000": "हज़ार",
    }

    # Number words in Gujarati
    GUJARATI_NUMBERS = {
        "0": "શૂન્ય",
        "1": "એક",
        "2": "બે",
        "3": "ત્રણ",
        "4": "ચાર",
        "5": "પાંચ",
        "6": "છ",
        "7": "સાત",
        "8": "આઠ",
        "9": "નવ",
        "10": "દસ",
        "100": "સો",
        "1000": "હજાર",
    }

    @staticmethod
    def normalize_numbers(text: str, lang: str = "hi") -> str:
        """Convert numbers to words"""
        # Handle special notation like {100}{एकसो}
        pattern = r"\{(\d+)\}\{([^}]+)\}"
        text = re.sub(pattern, r"\2", text)

        return text

    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """Normalize punctuation marks"""
        # Convert various quotes to standard
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)

        # Normalize dashes
        text = re.sub(r"[–—]", "-", text)

        return text

    @staticmethod
    def clean_text(text: str, lang: str = "hi") -> str:
        """Full text cleaning pipeline"""
        text = TextNormalizer.normalize_numbers(text, lang)
        text = TextNormalizer.normalize_punctuation(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
