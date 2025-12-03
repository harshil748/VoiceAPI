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
    """
    Tokenizer for TTS models - Compatible with SYSPIN VITS models

    IMPORTANT: For SYSPIN models, the vocabulary is defined by chars.txt file.
    The vocab structure is: [<PAD>] + list(chars.txt) + [<BLNK>]

    chars.txt contains ALL characters (including punctuation) in the EXACT
    order they were used during model training. DO NOT sort or reorder!
    """

    def __init__(self, config: CharactersConfig, use_chars_file_vocab: bool = False):
        self.config = config
        self.pad = config.pad
        self.eos = config.eos
        self.bos = config.bos
        self.blank = config.blank
        self.characters = config.characters
        self.punctuations = config.punctuations
        self.use_chars_file_vocab = use_chars_file_vocab

        # Build character to ID mapping
        self._build_vocab()

    def _build_vocab(self):
        """
        Build vocabulary from characters.

        For SYSPIN models (use_chars_file_vocab=True):
        - chars.txt contains ALL characters in EXACT order used during training
        - Vocab is simply: [<PAD>] + list(chars.txt) + [<BLNK>]
        - NO sorting, NO punctuation separation, NO reordering!

        The SYSPIN VitsCharacters._create_vocab() uses:
        self._vocab = [self._pad] + list(self._punctuations) + list(self._characters) + [self._blank]

        But when trained with graphemes=chars_txt and punctuations="", it becomes:
        self._vocab = [<PAD>] + [] + list(chars_txt) + [<BLNK>]
        = [<PAD>] + list(chars_txt) + [<BLNK>]
        """
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        idx = 0

        # 1. PAD token first
        if self.pad:
            self.char_to_id[self.pad] = idx
            self.id_to_char[idx] = self.pad
            idx += 1

        if self.use_chars_file_vocab:
            # SYSPIN model: chars.txt contains ALL chars in EXACT order
            # Add each character from chars.txt in order (no sorting!)
            for char in self.characters:
                if char not in self.char_to_id:
                    self.char_to_id[char] = idx
                    self.id_to_char[idx] = char
                    idx += 1
        else:
            # Legacy mode: punctuations first, then characters
            for char in self.punctuations:
                if char not in self.char_to_id:
                    self.char_to_id[char] = idx
                    self.id_to_char[idx] = char
                    idx += 1

            for char in self.characters:
                if char not in self.char_to_id:
                    self.char_to_id[char] = idx
                    self.id_to_char[idx] = char
                    idx += 1

        # BLANK token last
        if self.blank:
            self.char_to_id[self.blank] = idx
            self.id_to_char[idx] = self.blank
            idx += 1

        self.vocab_size = len(self.char_to_id)
        self.blank_id = self.char_to_id.get(self.blank)
        self.pad_id = self.char_to_id.get(self.pad)

    def text_to_ids(self, text: str, add_blank: bool = True) -> List[int]:
        """Convert text to token IDs with interspersed blanks"""
        text = self._clean_text(text)

        # First encode characters
        char_ids = []
        for char in text:
            if char in self.char_to_id:
                char_ids.append(self.char_to_id[char])

        # Intersperse blank tokens (matching VitsCharacters behavior)
        if add_blank and self.blank_id is not None:
            result = [self.blank_id] * (len(char_ids) * 2 + 1)
            result[1::2] = char_ids
            return result

        return char_ids

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
        """Text cleaning matching multilingual_cleaners"""
        # Apply multilingual cleaners pipeline
        text = text.lower()  # lowercase
        text = self._replace_symbols(text)
        text = self._remove_aux_symbols(text)
        text = re.sub(r"\s+", " ", text).strip()  # collapse_whitespace
        return text

    def _replace_symbols(self, text: str) -> str:
        """Replace symbols"""
        text = text.replace(";", ",")
        text = text.replace("-", " ")
        text = text.replace(":", ",")
        return text

    def _remove_aux_symbols(self, text: str) -> str:
        """Remove auxiliary symbols"""
        text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
        return text

    @classmethod
    def from_chars_file(cls, chars_file: str) -> "TTSTokenizer":
        """
        Create tokenizer from chars.txt file (for SYSPIN models)

        chars.txt contains ALL characters in the EXACT order used during training.
        The vocab will be: [<PAD>] + list(chars.txt) + [<BLNK>]
        """
        with open(chars_file, "r", encoding="utf-8") as f:
            chars = f.read().strip("\n")

        config = CharactersConfig(
            characters=chars,
            punctuations="",  # Empty - chars.txt already has everything
            pad="<PAD>",
            eos=None,  # VitsCharacters doesn't use EOS
            bos=None,  # VitsCharacters doesn't use BOS
            blank="<BLNK>",
        )

        return cls(config, use_chars_file_vocab=True)


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
