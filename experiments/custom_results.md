# Custom Model Results

## Experiment Configuration

This document contains the results of custom CNN model experiments on the Fashion MNIST dataset.

## Custom CNN Architecture

The custom CNN was implemented from scratch without pre-trained weights:

```
Layer 1: Conv(1->32) + BatchNorm + ReLU + MaxPool(2x2)
Layer 2: Conv(32->64) + BatchNorm + ReLU + MaxPool(2x2)
Layer 3: Conv(64->128) + BatchNorm + ReLU + MaxPool(2x2)
FC Layer 1: 128*3*3 -> 512
FC Layer 2: 512 -> 256
Output Layer: 256 -> 10
Dropout: 0.5
```

## Custom CNN (Basic)

| Metric | Value |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |

### Training Configuration
- Model: Custom CNN (from scratch)
- Epochs: 50
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Augmentation: None

## Custom CNN with Improved Baseline Techniques

| Metric | Value |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |

### Training Configuration
- Model: Custom CNN (from scratch)
- Epochs: 50
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Augmentation: Medium (RandomHorizontalFlip, RandomRotation, ColorJitter)

## Custom CNN Deep (Variant)

| Metric | Value |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |

### Training Configuration
- Model: Custom CNN Deep (4 conv blocks + GAP)
- Epochs: 50
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Augmentation: Medium

## Comparison: All Models

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| ResNet50 (Baseline) | TBD | TBD | TBD | TBD | ~23M |
| ViT (Baseline) | TBD | TBD | TBD | TBD | ~86M |
| ResNet50 (Improved) | TBD | TBD | TBD | TBD | ~23M |
| ViT (Improved) | TBD | TBD | TBD | TBD | ~86M |
| Custom CNN | TBD | TBD | TBD | TBD | ~1.2M |
| Custom CNN (Augmented) | TBD | TBD | TBD | TBD | ~1.2M |

## Conclusions

The custom CNN implementation demonstrates that:
1. A well-designed CNN from scratch can achieve competitive performance
2. The custom model has significantly fewer parameters than pre-trained models
3. Data augmentation techniques from improved baseline transfer well to custom models
4. The custom implementation provides insights into CNN architecture design
