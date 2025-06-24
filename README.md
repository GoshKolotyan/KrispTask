# Armenian ASR

Modern speech-to-text model for Armenian language using Wav2Vec2-BERT (see also with quantization and LoRA fine-tuning).

[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/GoshKolotyan/w2v-bert-2.0-armenian-CV16.0-version-demo)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

## Model Performance

Best model performance on Armenian Common Voice dataset:

| Metric | Score |
|--------|-------|
| Word Error Rate (WER) | 14.40% |
| Character Error Rate (CER) | 2.54% |
| Evaluation Loss | 0.1424 |

Pre-trained Model: [`GoshKolotyan/w2v-bert-2.0-armenian-CV16.0-version-demo`](https://huggingface.co/GoshKolotyan/w2v-bert-2.0-armenian-CV16.0-version-demo)

## Configuration

Edit `configs/configs.yml` for dataset and training settings.

For memory-efficient training, use `configs/quantize_configs.yml`.