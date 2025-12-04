# Training Datasets for Indian Language TTS

This directory contains training scripts and configurations for VITS-based TTS models.

## Datasets Used

### Primary Datasets (Non-SYSPIN/SPIRE)

| Dataset      | Language | Source                                           | License   | Samples | Duration |
| ------------ | -------- | ------------------------------------------------ | --------- | ------- | -------- |
| OpenSLR-103  | Hindi    | [OpenSLR](https://www.openslr.org/103/)          | CC BY 4.0 | ~10,000 | ~15 hrs  |
| OpenSLR-37   | Bengali  | [OpenSLR](https://www.openslr.org/37/)           | CC BY 4.0 | ~5,000  | ~8 hrs   |
| OpenSLR-64   | Marathi  | [OpenSLR](https://www.openslr.org/64/)           | CC BY 4.0 | ~3,000  | ~5 hrs   |
| OpenSLR-66   | Telugu   | [OpenSLR](https://www.openslr.org/66/)           | CC BY 4.0 | ~3,000  | ~5 hrs   |
| OpenSLR-79   | Kannada  | [OpenSLR](https://www.openslr.org/79/)           | CC BY 4.0 | ~3,000  | ~5 hrs   |
| OpenSLR-78   | Gujarati | [OpenSLR](https://www.openslr.org/78/)           | CC BY 4.0 | ~3,000  | ~5 hrs   |
| Common Voice | Hindi    | [Mozilla](https://commonvoice.mozilla.org/)      | CC0       | ~20,000 | ~25 hrs  |
| Common Voice | Bengali  | [Mozilla](https://commonvoice.mozilla.org/)      | CC0       | ~5,000  | ~8 hrs   |
| IndicTTS     | Multi    | [IIT Madras](https://www.iitm.ac.in/donlab/tts/) | Research  | ~50,000 | ~60 hrs  |

### Dataset Links (CSV Format)

```csv
Dataset Name,Language,URL,License,Type
OpenSLR Hindi ASR Corpus,Hindi,https://www.openslr.org/103/,CC BY 4.0,Speech Recognition
OpenSLR Bengali Multi-speaker,Bengali,https://www.openslr.org/37/,CC BY 4.0,Speech Recognition
OpenSLR Marathi,Marathi,https://www.openslr.org/64/,CC BY 4.0,Speech Recognition
OpenSLR Telugu,Telugu,https://www.openslr.org/66/,CC BY 4.0,Speech Recognition
OpenSLR Kannada,Kannada,https://www.openslr.org/79/,CC BY 4.0,Speech Recognition
OpenSLR Gujarati,Gujarati,https://www.openslr.org/78/,CC BY 4.0,Speech Recognition
Mozilla Common Voice Hindi,Hindi,https://commonvoice.mozilla.org/hi/datasets,CC0,Crowdsourced
Mozilla Common Voice Bengali,Bengali,https://commonvoice.mozilla.org/bn/datasets,CC0,Crowdsourced
IndicTTS Dataset,Multi,https://www.iitm.ac.in/donlab/tts/database.php,Research,TTS Corpus
Indic-Voices,Multi,https://ai4bharat.iitm.ac.in/indic-voices/,CC BY 4.0,Multilingual Speech
Google FLEURS,Multi,https://huggingface.co/datasets/google/fleurs,CC BY 4.0,Multilingual NLU
```

## Training Pipeline

### 1. Data Preparation

```bash
# Download and prepare Hindi dataset
python prepare_dataset.py \
    --input /path/to/openslr_hindi \
    --output data/hindi_female \
    --language hindi \
    --format openslr \
    --split

# Prepare Common Voice data
python prepare_dataset.py \
    --input /path/to/commonvoice_hindi \
    --output data/hindi_cv \
    --language hindi \
    --format commonvoice \
    --split
```

### 2. Training

```bash
# Train Hindi female voice
python train_vits.py \
    --config configs/hindi_female.yaml \
    --data data/hindi_female \
    --output output/hindi_female \
    --language hindi \
    --gender female

# Resume training from checkpoint
python train_vits.py \
    --config configs/hindi_female.yaml \
    --data data/hindi_female \
    --output output/hindi_female \
    --resume output/hindi_female/checkpoints/checkpoint_100000.pth
```

### 3. Export Model

```bash
# Export to JIT format for inference
python export_model.py \
    --checkpoint output/hindi_female/checkpoints/best_model.pth \
    --output models/hi_female/hi_female_vits_30hrs.pt \
    --format jit
```

## Model Architecture

- **Base Model**: VITS (Variational Inference with adversarial learning for Text-to-Speech)
- **Encoder**: Transformer-based text encoder with 6 layers
- **Decoder**: HiFi-GAN based neural vocoder
- **Duration Predictor**: Stochastic duration predictor
- **Sample Rate**: 22050 Hz
- **Mel Channels**: 80

## Training Hyperparameters

| Parameter           | Value       |
| ------------------- | ----------- |
| Learning Rate       | 2e-4        |
| Batch Size          | 32          |
| Epochs              | 1000        |
| Warmup Epochs       | 50          |
| Checkpoint Interval | 10000 steps |
| FP16 Training       | Yes         |
| Optimizer           | AdamW       |

## Hardware Requirements

- **GPU**: NVIDIA V100/A100 (16GB+ VRAM)
- **RAM**: 32GB+
- **Storage**: 100GB+ for datasets and checkpoints
- **Training Time**: ~48-72 hours per language on V100

## License

Training scripts are released under MIT License.
Individual datasets retain their original licenses (see table above).
