# LoRA Fine-Tuning for Text Classification

Fine-tuning RoBERTa on AG News using LoRA (Low-Rank Adaptation) for parameter-efficient training.

## Results
- **Accuracy:** 94.3%
- **Trainable Parameters:** 728K out of 125M (0.58%)
- **Training Time:** ~2 hours on A100 GPU

## What is LoRA?

LoRA freezes the base model and only trains small matrices injected into attention layers. This reduces trainable parameters by 99% while maintaining performance.

## Key Features
- Data cleaning and deduplication
- Text augmentation (20% of training data)
- Weighted loss for class imbalance
- Early stopping with cosine learning rate scheduling

## Tech Stack
- **Model:** RoBERTa-base
- **Framework:** HuggingFace Transformers, PEFT
- **Dataset:** AG News (4-class news classification)

## Quick Start
```bash
pip install transformers datasets peft torch
python Deep_learning_project_2.ipynb
```

## Architecture
- **LoRA rank:** 16
- **LoRA alpha:** 32
- **Target modules:** Query attention (layers 9-11) + Dense (layer 11)
- **Max sequence length:** 192
- **Batch size:** 16

## Results by Class

| Class | F1-Score |
|-------|----------|
| World | 95.4% |
| Sports | 98.8% |
| Business | 91.4% |
| Sci/Tech | 91.8% |

## References
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [RoBERTa](https://arxiv.org/abs/1907.11692) - Liu et al., 2019
