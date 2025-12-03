import os
from extra import TTSTokenizer, VitsConfig, CharactersConfig, VitsCharacters
import torch
import numpy as np

#ch female
letters="खछगचऊुलणशढ़इौज़झठैढजफ़औ्ड़फूेानटॅयव़ऋदप.थअँऑआघहतषरसभउञडएईऐक़ िओ?धी,ॉंख़कोबमृ"
model="ch_male_vits_30hrs.pt"
text = "पेरिविंकल के जड़, उपजी अउ पत्त्ता मन ह बिकट उपयोगी हे"

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