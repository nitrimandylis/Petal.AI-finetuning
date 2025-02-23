# Petal.AI LLM Fine-tuning

This repository contains the Python scripts and utilities used to fine-tune Large Language Models (LLMs) for deployment in the Petal.AI application. The codebase provides a complete pipeline for data preprocessing, model fine-tuning, and conversion to various deployment formats.

## Project Structure

```
├── Datasets/
│   ├── skills.csv         # Original training dataset
│   └── skills_edited.csv  # Processed dataset--not recommended
├── Python scripts/
│   ├── clean_csv.py           # Data preprocessing utility
│   ├── train_model.py         # Model fine-tuning script
│   ├── test_model.py         # Model testing utility
│   ├── convert_model_ggml.py  # GGML format converter
│   └── convert_to_coreml.py   # CoreML format converter
```

## Features

- Data preprocessing and cleaning
- Model fine-tuning with customizable parameters
- Multiple deployment format conversions (CoreML, GGML)
- Testing utilities for model evaluation

## Prerequisites

- Python 3.x
- PyTorch
- Transformers library
- CoreMLTools (for iOS deployment)
- SQLite (for data management)

## Model Configuration

The fine-tuning process uses the following key parameters:
- Base model: GPT-2
- Training epochs: 19
- Batch size: 8
- Learning rate: 2e-4
- Gradient accumulation steps: 8
- Warmup ratio: 0.03

## Deployment

The fine-tuned models can be converted to:
- CoreML format for iOS deployment
- GGML format for efficient inference
- SafeTensors format for secure model storage

## License

MIT License

