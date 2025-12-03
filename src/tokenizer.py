"""
TTS Tokenizer for VITS models
Adapted from Coqui TTS for SYSPIN models

CRITICAL: The vocabulary MUST be built as:
[<PAD>] + list(punctuations) + list(characters) + [<BLNK>]

Where:
- punctuations = "!¡'(),-.:;¿? " (standard VITS punctuations)
- characters = content of chars.txt file
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass


# Standard VITS punctuations used by SYSPIN models
VITS_PUNCTUATIONS = "!¡'(),-.:;¿? "


@dataclass
class CharactersConfig:
    """Character configuration for tokenizer"""

    characters: str = ""
    punctuations: str = VITS_PUNCTUATIONS
    pad: str = "<PAD>"
    eos: str = None
    bos: str = None
    blank: str = "<BLNK>"
    phonemes: Optional[str] = None


class TTSTokenizer:
    """
    Tokenizer for TTS models - Compatible with SYSPIN VITS models.

    The vocabulary is built EXACTLY as VitsCharacters._create_vocab():
    vocab = [<PAD>] + list(punctuations) + list(characters) + [<BLNK>]

    For SYSPIN models:
    - punctuations = "!¡'(),-.:;¿? " (13 chars)
    - characters = content from chars.txt
    - Total vocab = 1 + 13 + len(chars.txt) + 1
    """

    def __init__(
        self,
        characters: str,
        punctuations: str = VITS_PUNCTUATIONS,
        pad: str = "<PAD>",
        blank: str = "<BLNK>",
    ):
        """
        Initialize tokenizer.

        Args:
            characters: The characters string (from chars.txt)
            punctuations: Punctuation characters (default: VITS standard)
            pad: Padding token
            blank: Blank token for CTC
        """
        self.characters = characters
        self.punctuations = punctuations
        self.pad = pad
        self.blank = blank

        # Build vocabulary: [PAD] + punctuations + characters + [BLANK]
        self._build_vocab()

    def _build_vocab(self):
        """
        Build vocabulary EXACTLY matching VitsCharacters._create_vocab():
        self._vocab = [self._pad] + list(self._punctuations) + list(self._characters) + [self._blank]
        """
        self.vocab: List[str] = []
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        # Build vocab in exact order
        # 1. PAD token
        self.vocab.append(self.pad)

        # 2. Punctuations
        for char in self.punctuations:
            self.vocab.append(char)

        # 3. Characters from chars.txt
        for char in self.characters:
            self.vocab.append(char)

        # 4. BLANK token
        self.vocab.append(self.blank)

        # Build mappings
        for idx, char in enumerate(self.vocab):
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char

        self.vocab_size = len(self.vocab)
        self.pad_id = self.char_to_id[self.pad]
        self.blank_id = self.char_to_id[self.blank]

    def text_to_ids(self, text: str, add_blank: bool = True) -> List[int]:
        """
        Convert text to token IDs with interspersed blanks.

        Matches TTSTokenizer.text_to_ids() from extra.py:
        1. Clean text with multilingual_cleaners
        2. Encode to IDs
        3. Intersperse blank tokens
        """
        # Apply multilingual_cleaners
        text = self._clean_text(text)

        # Encode characters to IDs
        char_ids = []
        for char in text:
            if char in self.char_to_id:
                char_ids.append(self.char_to_id[char])
            # Skip unknown characters (matching original behavior)

        # Intersperse blank tokens
        if add_blank:
            result = [self.blank_id] * (len(char_ids) * 2 + 1)
            result[1::2] = char_ids
            return result

        return char_ids

    def ids_to_text(self, ids: List[int]) -> str:
        """Convert token IDs back to text"""
        chars = []
        for idx in ids:
            if idx in self.id_to_char:
                char = self.id_to_char[idx]
                if char not in [self.pad, self.blank]:
                    chars.append(char)
        return "".join(chars)

    def _clean_text(self, text: str) -> str:
        """
        Text cleaning matching multilingual_cleaners from extra.py:
        1. lowercase
        2. replace_symbols
        3. remove_aux_symbols
        4. collapse_whitespace
        """
        text = text.lower()
        text = self._replace_symbols(text)
        text = self._remove_aux_symbols(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _replace_symbols(self, text: str) -> str:
        """Replace symbols matching extra.py replace_symbols()"""
        text = text.replace(";", ",")
        text = text.replace("-", " ")
        text = text.replace(":", ",")
        return text

    def _remove_aux_symbols(self, text: str) -> str:
        """Remove auxiliary symbols matching extra.py remove_aux_symbols()"""
        text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
        return text

    @classmethod
    def from_chars_file(cls, chars_file: str) -> "TTSTokenizer":
        """
        Create tokenizer from chars.txt file.

        This matches the jit_infer.py setup:
        - characters = content of chars.txt
        - punctuations = "!¡'(),-.:;¿? " (standard VITS punctuations)

        Vocab will be: [<PAD>] + punctuations + characters + [<BLNK>]
        """
        with open(chars_file, "r", encoding="utf-8") as f:
            characters = f.read().strip("\n")

        return cls(
            characters=characters,
            punctuations=VITS_PUNCTUATIONS,
            pad="<PAD>",
            blank="<BLNK>",
        )


class TextNormalizer:
    """Text normalizer for Indian languages"""

    @staticmethod
    def normalize_numbers(text: str, lang: str = "hi") -> str:
        """Convert numbers to words"""
        pattern = r"\{(\d+)\}\{([^}]+)\}"
        text = re.sub(pattern, r"\2", text)
        return text

    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """Normalize punctuation marks"""
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        text = re.sub(r"[–—]", "-", text)
        return text

    @staticmethod
    def clean_text(text: str, lang: str = "hi") -> str:
        """Full text cleaning pipeline"""
        text = TextNormalizer.normalize_numbers(text, lang)
        text = TextNormalizer.normalize_punctuation(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
