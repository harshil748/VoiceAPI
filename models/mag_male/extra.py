from typing import Callable, Dict, List, Union
from dataclasses import asdict, dataclass, field


import re
from dataclasses import replace
from typing import Dict
_whitespace_re = re.compile(r"\s+")

from dataclasses import dataclass, field
from typing import List

# from TTS.tts.configs.shared_configs import BaseTTSConfig
# from TTS.tts.models.vits import VitsArgs, VitsAudioConfig

@dataclass
class CharactersConfig():

    characters_class: str = None

    # using BaseVocabulary
    vocab_dict: Dict = None

    # using on BaseCharacters
    pad: str = None
    eos: str = None
    bos: str = None
    blank: str = None
    characters: str = None
    punctuations: str = None
    phonemes: str = None
    is_unique: bool = True  # for backwards compatibility of models trained with char sets with duplicates
    is_sorted: bool = True


@dataclass
class BaseTTSConfig():

    # audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    # phoneme settings
    use_phonemes: bool = False
    phonemizer: str = None
    phoneme_language: str = None
    compute_input_seq_cache: bool = False
    text_cleaner: str = None
    enable_eos_bos_chars: bool = False
    test_sentences_file: str = ""
    phoneme_cache_path: str = None
    # vocabulary parameters
    characters: CharactersConfig = None
    add_blank: bool = False
    # training params
    batch_group_size: int = 0
    loss_masking: bool = None
    # dataloading
    min_audio_len: int = 1
    max_audio_len: int = float("inf")
    min_text_len: int = 1
    max_text_len: int = float("inf")
    compute_f0: bool = False
    compute_energy: bool = False
    compute_linear_spec: bool = False
    precompute_num_workers: int = 0
    use_noise_augment: bool = False
    start_by_longest: bool = False
    shuffle: bool = False
    drop_last: bool = False
    # dataset
    datasets: str = None
    # optimizer
    optimizer: str = "radam"
    optimizer_params: dict = None
    # scheduler
    lr_scheduler: str = None
    lr_scheduler_params: dict = field(default_factory=lambda: {})
    # testing
    test_sentences: List[str] = field(default_factory=lambda: [])
    # evaluation
    eval_split_max_size: int = None
    eval_split_size: float = 0.01
    # weighted samplers
    use_speaker_weighted_sampler: bool = False
    speaker_weighted_sampler_alpha: float = 1.0
    use_language_weighted_sampler: bool = False
    language_weighted_sampler_alpha: float = 1.0
    use_length_weighted_sampler: bool = False
    length_weighted_sampler_alpha: float = 1.0


@dataclass
class VitsAudioConfig():
    fft_size: int = 1024
    sample_rate: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    num_mels: int = 80
    mel_fmin: int = 0
    mel_fmax: int = None
    
@dataclass
class VitsArgs():
    num_chars: int = 100
    out_channels: int = 513
    spec_segment_size: int = 32
    hidden_channels: int = 192
    hidden_channels_ffn_text_encoder: int = 768
    num_heads_text_encoder: int = 2
    num_layers_text_encoder: int = 6
    kernel_size_text_encoder: int = 3
    dropout_p_text_encoder: float = 0.1
    dropout_p_duration_predictor: float = 0.5
    kernel_size_posterior_encoder: int = 5
    dilation_rate_posterior_encoder: int = 1
    num_layers_posterior_encoder: int = 16
    kernel_size_flow: int = 5
    dilation_rate_flow: int = 1
    num_layers_flow: int = 4
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates_decoder: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    periods_multi_period_discriminator: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    use_sdp: bool = True
    noise_scale: float = 1.0
    inference_noise_scale: float = 0.667
    length_scale: float = 1
    noise_scale_dp: float = 1.0
    inference_noise_scale_dp: float = 1.0
    max_inference_len: int = None
    init_discriminator: bool = True
    use_spectral_norm_disriminator: bool = False
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speakers_file: str = None
    d_vector_file: List[str] = None
    speaker_embedding_channels: int = 256
    use_d_vector_file: bool = False
    d_vector_dim: int = 0
    detach_dp_input: bool = True
    use_language_embedding: bool = False
    embedded_language_dim: int = 4
    num_languages: int = 0
    language_ids_file: str = None
    use_speaker_encoder_as_loss: bool = False
    speaker_encoder_config_path: str = ""
    speaker_encoder_model_path: str = ""
    condition_dp_on_speaker: bool = True
    freeze_encoder: bool = False
    freeze_DP: bool = False
    freeze_PE: bool = False
    freeze_flow_decoder: bool = False
    freeze_waveform_decoder: bool = False
    encoder_sample_rate: int = None
    interpolate_z: bool = True
    reinit_DP: bool = False
    reinit_text_encoder: bool = False
@dataclass
class VitsConfig(BaseTTSConfig):

    model: str = "vits"
    # model specific params
    model_args: VitsArgs = field(default_factory=VitsArgs)
    audio: VitsAudioConfig = field(default_factory=VitsAudioConfig)

    # optimizer
    grad_clip: List[float] = field(default_factory=lambda: [1000, 1000])
    lr_gen: float = 0.0002
    lr_disc: float = 0.0002
    lr_scheduler_gen: str = "ExponentialLR"
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    lr_scheduler_disc: str = "ExponentialLR"
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    scheduler_after_epoch: bool = True
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01})

    # loss params
    kl_loss_alpha: float = 1.0
    disc_loss_alpha: float = 1.0
    gen_loss_alpha: float = 1.0
    feat_loss_alpha: float = 1.0
    mel_loss_alpha: float = 45.0
    dur_loss_alpha: float = 1.0
    speaker_encoder_loss_alpha: float = 1.0

    # data loader params
    return_wav: bool = True
    compute_linear_spec: bool = True

    # sampler params
    use_weighted_sampler: bool = False  # TODO: move it to the base config
    weighted_sampler_attrs: dict = field(default_factory=lambda: {})
    weighted_sampler_multipliers: dict = field(default_factory=lambda: {})

    # overrides
    r: int = 1  # DO NOT CHANGE
    add_blank: bool = True

    # testing
    test_sentences: List[List] = field(
        default_factory=lambda: [
            ["It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."],
            ["Be a voice, not an echo."],
            ["I'm sorry Dave. I'm afraid I can't do that."],
            ["This cake is great. It's so delicious and moist."],
            ["Prior to November 22, 1963."],
        ]
    )

    # multi-speaker settings
    # use speaker embedding layer
    num_speakers: int = 0
    use_speaker_embedding: bool = False
    speakers_file: str = None
    speaker_embedding_channels: int = 256
    language_ids_file: str = None
    use_language_embedding: bool = False

    # use d-vectors
    use_d_vector_file: bool = False
    d_vector_file: List[str] = None
    d_vector_dim: int = None

    def __post_init__(self):
        pass
        # for key, val in self.model_args.items():
        #     if hasattr(self, key):
        #         self[key] = val





def parse_symbols():
    return {
        "pad": _pad,
        "eos": _eos,
        "bos": _bos,
        "characters": _characters,
        "punctuations": _punctuations,
        "phonemes": _phonemes,
    }


# DEFAULT SET OF GRAPHEMES
_pad = "<PAD>"
_eos = "<EOS>"
_bos = "<BOS>"
_blank = "<BLNK>"  # TODO: check if we need this alongside with PAD
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_punctuations = "!'(),-.:;? "


# DEFAULT SET OF IPA PHONEMES
# Phonemes definition (All IPA characters)
_vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
_non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
_pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
_suprasegmentals = "ˈˌːˑ"
_other_symbols = "ʍwɥʜʢʡɕʑɺɧʲ"
_diacrilics = "ɚ˞ɫ"
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics


class BaseVocabulary:
    """Base Vocabulary class.

    This class only needs a vocabulary dictionary without specifying the characters.

    Args:
        vocab (Dict): A dictionary of characters and their corresponding indices.
    """

    def __init__(self, vocab: Dict, pad: str = None, blank: str = None, bos: str = None, eos: str = None):
        self.vocab = vocab
        self.pad = pad
        self.blank = blank
        self.bos = bos
        self.eos = eos

    @property
    def pad_id(self) -> int:
        """Return the index of the padding character. If the padding character is not specified, return the length
        of the vocabulary."""
        return self.char_to_id(self.pad) if self.pad else len(self.vocab)

    @property
    def blank_id(self) -> int:
        """Return the index of the blank character. If the blank character is not specified, return the length of
        the vocabulary."""
        return self.char_to_id(self.blank) if self.blank else len(self.vocab)

    @property
    def bos_id(self) -> int:
        """Return the index of the bos character. If the bos character is not specified, return the length of the
        vocabulary."""
        return self.char_to_id(self.bos) if self.bos else len(self.vocab)

    @property
    def eos_id(self) -> int:
        """Return the index of the eos character. If the eos character is not specified, return the length of the
        vocabulary."""
        return self.char_to_id(self.eos) if self.eos else len(self.vocab)

    @property
    def vocab(self):
        """Return the vocabulary dictionary."""
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        """Set the vocabulary dictionary and character mapping dictionaries."""
        self._vocab, self._char_to_id, self._id_to_char = None, None, None
        if vocab is not None:
            self._vocab = vocab
            self._char_to_id = {char: idx for idx, char in enumerate(self._vocab)}
            self._id_to_char = {
                idx: char for idx, char in enumerate(self._vocab)  # pylint: disable=unnecessary-comprehension
            }

    @staticmethod
    def init_from_config(config, **kwargs):
        """Initialize from the given config."""
        if config.characters is not None and "vocab_dict" in config.characters and config.characters.vocab_dict:
            return (
                BaseVocabulary(
                    config.characters.vocab_dict,
                    config.characters.pad,
                    config.characters.blank,
                    config.characters.bos,
                    config.characters.eos,
                ),
                config,
            )
        return BaseVocabulary(**kwargs), config

    def to_config(self):
        return CharactersConfig(
            vocab_dict=self._vocab,
            pad=self.pad,
            eos=self.eos,
            bos=self.bos,
            blank=self.blank,
            is_unique=False,
            is_sorted=False,
        )

    @property
    def num_chars(self):
        """Return number of tokens in the vocabulary."""
        return len(self._vocab)

    def char_to_id(self, char: str) -> int:
        """Map a character to an token ID."""
        try:
            return self._char_to_id[char]
        except KeyError as e:
            raise KeyError(f" [!] {repr(char)} is not in the vocabulary.") from e

    def id_to_char(self, idx: int) -> str:
        """Map an token ID to a character."""
        return self._id_to_char[idx]


class BaseCharacters:


    def __init__(
        self,
        characters: str = None,
        punctuations: str = None,
        pad: str = None,
        eos: str = None,
        bos: str = None,
        blank: str = None,
        is_unique: bool = False,
        is_sorted: bool = True,
    ) -> None:
        self._characters = characters
        self._punctuations = punctuations
        self._pad = pad
        self._eos = eos
        self._bos = bos
        self._blank = blank
        self.is_unique = is_unique
        self.is_sorted = is_sorted
        self._create_vocab()

    @property
    def pad_id(self) -> int:
        return self.char_to_id(self.pad) if self.pad else len(self.vocab)

    @property
    def blank_id(self) -> int:
        return self.char_to_id(self.blank) if self.blank else len(self.vocab)

    @property
    def eos_id(self) -> int:
        return self.char_to_id(self.eos) if self.eos else len(self.vocab)

    @property
    def bos_id(self) -> int:
        return self.char_to_id(self.bos) if self.bos else len(self.vocab)

    @property
    def characters(self):
        return self._characters

    @characters.setter
    def characters(self, characters):
        self._characters = characters
        self._create_vocab()

    @property
    def punctuations(self):
        return self._punctuations

    @punctuations.setter
    def punctuations(self, punctuations):
        self._punctuations = punctuations
        self._create_vocab()

    @property
    def pad(self):
        return self._pad

    @pad.setter
    def pad(self, pad):
        self._pad = pad
        self._create_vocab()

    @property
    def eos(self):
        return self._eos

    @eos.setter
    def eos(self, eos):
        self._eos = eos
        self._create_vocab()

    @property
    def bos(self):
        return self._bos

    @bos.setter
    def bos(self, bos):
        self._bos = bos
        self._create_vocab()

    @property
    def blank(self):
        return self._blank

    @blank.setter
    def blank(self, blank):
        self._blank = blank
        self._create_vocab()

    @property
    def vocab(self):
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        self._vocab = vocab
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self._id_to_char = {
            idx: char for idx, char in enumerate(self.vocab)  # pylint: disable=unnecessary-comprehension
        }

    @property
    def num_chars(self):
        return len(self._vocab)

    def _create_vocab(self):
        _vocab = self._characters
        if self.is_unique:
            _vocab = list(set(_vocab))
        if self.is_sorted:
            _vocab = sorted(_vocab)
        _vocab = list(_vocab)
        _vocab = [self._blank] + _vocab if self._blank is not None and len(self._blank) > 0 else _vocab
        _vocab = [self._bos] + _vocab if self._bos is not None and len(self._bos) > 0 else _vocab
        _vocab = [self._eos] + _vocab if self._eos is not None and len(self._eos) > 0 else _vocab
        _vocab = [self._pad] + _vocab if self._pad is not None and len(self._pad) > 0 else _vocab
        self.vocab = _vocab + list(self._punctuations)
        if self.is_unique:
            duplicates = {x for x in self.vocab if self.vocab.count(x) > 1}
            assert (
                len(self.vocab) == len(self._char_to_id) == len(self._id_to_char)
            ), f" [!] There are duplicate characters in the character set. {duplicates}"

    def char_to_id(self, char: str) -> int:
        try:
            return self._char_to_id[char]
        except KeyError as e:
            raise KeyError(f" [!] {repr(char)} is not in the vocabulary.") from e

    def id_to_char(self, idx: int) -> str:
        return self._id_to_char[idx]

    def print_log(self, level: int = 0):
        """
        Prints the vocabulary in a nice format.
        """
        indent = "\t" * level
        print(f"{indent}| > Characters: {self._characters}")
        print(f"{indent}| > Punctuations: {self._punctuations}")
        print(f"{indent}| > Pad: {self._pad}")
        print(f"{indent}| > EOS: {self._eos}")
        print(f"{indent}| > BOS: {self._bos}")
        print(f"{indent}| > Blank: {self._blank}")
        print(f"{indent}| > Vocab: {self.vocab}")
        print(f"{indent}| > Num chars: {self.num_chars}")

    @staticmethod
    def init_from_config(config: "Coqpit"):  # pylint: disable=unused-argument
        """Init your character class from a config.

        Implement this method for your subclass.
        """
        # use character set from config
        if config.characters is not None:
            return BaseCharacters(**config.characters), config
        # return default character set
        characters = BaseCharacters()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config

    def to_config(self) -> "CharactersConfig":
        return CharactersConfig(
            characters=self._characters,
            punctuations=self._punctuations,
            pad=self._pad,
            eos=self._eos,
            bos=self._bos,
            blank=self._blank,
            is_unique=self.is_unique,
            is_sorted=self.is_sorted,
        )


class IPAPhonemes(BaseCharacters):
 

    def __init__(
        self,
        characters: str = _phonemes,
        punctuations: str = _punctuations,
        pad: str = _pad,
        eos: str = _eos,
        bos: str = _bos,
        blank: str = _blank,
        is_unique: bool = False,
        is_sorted: bool = True,
    ) -> None:
        super().__init__(characters, punctuations, pad, eos, bos, blank, is_unique, is_sorted)

    @staticmethod
    def init_from_config(config: "Coqpit"):
        """Init a IPAPhonemes object from a model config

        If characters are not defined in the config, it will be set to the default characters and the config
        will be updated.
        """
        # band-aid for compatibility with old models
        if "characters" in config and config.characters is not None:
            if "phonemes" in config.characters and config.characters.phonemes is not None:
                config.characters["characters"] = config.characters["phonemes"]
            return (
                IPAPhonemes(
                    characters=config.characters["characters"],
                    punctuations=config.characters["punctuations"],
                    pad=config.characters["pad"],
                    eos=config.characters["eos"],
                    bos=config.characters["bos"],
                    blank=config.characters["blank"],
                    is_unique=config.characters["is_unique"],
                    is_sorted=config.characters["is_sorted"],
                ),
                config,
            )
        # use character set from config
        if config.characters is not None:
            return IPAPhonemes(**config.characters), config
        # return default character set
        characters = IPAPhonemes()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config


class Graphemes(BaseCharacters):
 

    def __init__(
        self,
        characters: str = _characters,
        punctuations: str = _punctuations,
        pad: str = _pad,
        eos: str = _eos,
        bos: str = _bos,
        blank: str = _blank,
        is_unique: bool = False,
        is_sorted: bool = True,
    ) -> None:
        super().__init__(characters, punctuations, pad, eos, bos, blank, is_unique, is_sorted)

    @staticmethod
    def init_from_config(config: "Coqpit"):
        """Init a Graphemes object from a model config

        If characters are not defined in the config, it will be set to the default characters and the config
        will be updated.
        """
        if config.characters is not None:
            # band-aid for compatibility with old models
            if "phonemes" in config.characters:
                return (
                    Graphemes(
                        characters=config.characters["characters"],
                        punctuations=config.characters["punctuations"],
                        pad=config.characters["pad"],
                        eos=config.characters["eos"],
                        bos=config.characters["bos"],
                        blank=config.characters["blank"],
                        is_unique=config.characters["is_unique"],
                        is_sorted=config.characters["is_sorted"],
                    ),
                    config,
                )
            return Graphemes(**config.characters), config
        characters = Graphemes()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config


if __name__ == "__main__":
    gr = Graphemes()
    ph = IPAPhonemes()
    gr.print_log()
    ph.print_log()


class VitsCharacters(BaseCharacters):
    """Characters class for VITs model for compatibility with pre-trained models"""

    def __init__(
        self,
        graphemes: str = _characters,
        punctuations: str = _punctuations,
        pad: str = _pad,
        ipa_characters: str = _phonemes,
    ) -> None:
        if ipa_characters is not None:
            graphemes += ipa_characters
        super().__init__(graphemes, punctuations, pad, None, None, "<BLNK>", is_unique=False, is_sorted=True)

    def _create_vocab(self):
        self._vocab = [self._pad] + list(self._punctuations) + list(self._characters) + [self._blank]
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        # pylint: disable=unnecessary-comprehension
        self._id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    @staticmethod
    def init_from_config(config):
        _pad = config.characters.pad
        _punctuations = config.characters.punctuations
        _letters = config.characters.characters
        _letters_ipa = config.characters.phonemes
        return (
            VitsCharacters(graphemes=_letters, ipa_characters=_letters_ipa, punctuations=_punctuations, pad=_pad),
            config,
        )

    def to_config(self) -> "CharactersConfig":
        return CharactersConfig(
            characters=self._characters,
            punctuations=self._punctuations,
            pad=self._pad,
            eos=None,
            bos=None,
            blank=self._blank,
            is_unique=False,
            is_sorted=True,
        )
        
class TTSTokenizer:
    def __init__(
        self,
        text_cleaner: Callable = None,
        characters: "BaseCharacters" = None,
    ):
        self.text_cleaner = text_cleaner
        self.characters = characters
        self.not_found_characters = []

    @property
    def characters(self):
        return self._characters

    @characters.setter
    def characters(self, new_characters):
        self._characters = new_characters
        self.pad_id = self.characters.char_to_id(self.characters.pad) if self.characters.pad else None
        self.blank_id = self.characters.char_to_id(self.characters.blank) if self.characters.blank else None

    def encode(self, text: str) -> List[int]:
        """Encodes a string of text as a sequence of IDs."""
        token_ids = []
        for char in text:
            try:
                idx = self.characters.char_to_id(char)
                token_ids.append(idx)
            except KeyError:
                # discard but store not found characters
                if char not in self.not_found_characters:
                    self.not_found_characters.append(char)
                    print(text)
                    print(f" [!] Character {repr(char)} not found in the vocabulary. Discarding it.")
        return token_ids

    def text_to_ids(self, text: str, language: str = None) -> List[int]:  # pylint: disable=unused-argument
        text = self.text_cleaner(text)
        text = self.encode(text)
        text = self.intersperse_blank_char(text, True)
        return text

    def pad_with_bos_eos(self, char_sequence: List[str]):
        """Pads a sequence with the special BOS and EOS characters."""
        return [self.characters.bos_id] + list(char_sequence) + [self.characters.eos_id]

    def intersperse_blank_char(self, char_sequence: List[str], use_blank_char: bool = False):
        """Intersperses the blank character between characters in a sequence.

        Use the ```blank``` character if defined else use the ```pad``` character.
        """
        char_to_use = self.characters.blank_id if use_blank_char else self.characters.pad
        result = [char_to_use] * (len(char_sequence) * 2 + 1)
        result[1::2] = char_sequence
        return result

    @staticmethod
    def init_from_config(config: "Coqpit", characters: "BaseCharacters" = None):
        text_cleaner = multilingual_cleaners
        CharactersClass = VitsCharacters
        characters, new_config = CharactersClass.init_from_config(config)
        # new_config.characters.characters_class = get_import_path(characters)
        new_config.characters.characters_class = VitsCharacters
        return (
            TTSTokenizer(text_cleaner, characters),new_config)


def multilingual_cleaners(text):
    """Pipeline for multilingual text"""
    text = lowercase(text)
    text = replace_symbols(text, lang=None)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()

def replace_symbols(text, lang="en"):

    text = text.replace(";", ",")
    text = text.replace("-", " ") if lang != "ca" else text.replace("-", "")
    text = text.replace(":", ",")
    if lang == "en":
        text = text.replace("&", " and ")
    elif lang == "fr":
        text = text.replace("&", " et ")
    elif lang == "pt":
        text = text.replace("&", " e ")
    elif lang == "ca":
        text = text.replace("&", " i ")
        text = text.replace("'", "")
    return text

def remove_aux_symbols(text):
    text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
    return text