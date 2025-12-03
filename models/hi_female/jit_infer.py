import os
from extra import TTSTokenizer, VitsConfig, CharactersConfig, VitsCharacters
import torch
import numpy as np

#ch female
with open("chars.txt", 'r') as f:
    letters = f.read().strip('\n')
model="hi_female_vits_30hrs.pt"
text = "फिल्म गर्दिश में अमरीश पुरी के साथ जैकी श्रॉफ, ऐश्वर्या, डिंपल कपाड़िया"

config = VitsConfig(
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
        characters_class=VitsCharacters,
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=letters,
        punctuations="!¡'(),-.:;¿? ",
        phonemes=None)
    )
tokenizer, config = TTSTokenizer.init_from_config(config)

x = tokenizer.text_to_ids(text)
x = torch.from_numpy(np.array(x)).unsqueeze(0)
net = torch.jit.load(model)
with torch.no_grad():
    out2 = net(x)
import soundfile as sf
sf.write("jit.wav", out2.squeeze().cpu().numpy(), 22050)