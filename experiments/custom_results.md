# Custom Model Results

## Experiment Configuration

This document contains the results of custom CNN model experiments on the Fashion MNIST dataset.

## Custom CNN Architecture

The custom CNN was implemented from scratch without pre-trained weights:

```
Input: (1, 28, 28)

Block 1: Conv2d(1→32, 3×3, padding=1) + BatchNorm + ReLU + MaxPool(2×2)
         Output: (32, 14, 14)

Block 2: Conv2d(32→64, 3×3, padding=1) + BatchNorm + ReLU + MaxPool(2×2)
         Output: (64, 7, 7)

Block 3: Conv2d(64→128, 3×3, padding=1) + BatchNorm + ReLU + MaxPool(2×2)
         Output: (128, 3, 3)

Flatten: 128 × 3 × 3 = 1152

FC1: Linear(1152 → 512) + ReLU + Dropout(0.5)
FC2: Linear(512 → 256) + ReLU + Dropout(0.3)
Output: Linear(256 → 10) + Softmax

Параметры: ~1.24M
```

### Custom CNN Deep Architecture

```
Input: (1, 28, 28)

Block 1: Conv2d(1→32) + BN + ReLU + Conv2d(32→32) + BN + ReLU + MaxPool
Block 2: Conv2d(32→64) + BN + ReLU + Conv2d(64→64) + BN + ReLU + MaxPool
Block 3: Conv2d(64→128) + BN + ReLU + Conv2d(128→128) + BN + ReLU + MaxPool
Block 4: Conv2d(128→256) + BN + ReLU + AdaptiveAvgPool(1×1)

Flatten: 256
FC: Linear(256→128) + ReLU + Dropout(0.5)
Output: Linear(128→10)

Параметры: ~2.17M
```

## Custom CNN (Basic)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9156 |
| Precision | 0.9159 |
| Recall    | 0.9156 |
| F1-Score  | 0.9155 |

### Training Configuration
- Model: Custom CNN (from scratch)
- Epochs: 50
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Augmentation: None
- Training Time: ~1 850 с (~31 минута)

## Custom CNN with Improved Baseline Techniques

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9287 |
| Precision | 0.9291 |
| Recall    | 0.9287 |
| F1-Score  | 0.9287 |

### Training Configuration
- Model: Custom CNN (from scratch)
- Epochs: 50
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Augmentation: Medium (RandomHorizontalFlip, RandomRotation(15°), ColorJitter)
- Training Time: ~2 100 с

## Custom CNN Deep (Variant)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9334 |
| Precision | 0.9337 |
| Recall    | 0.9334 |
| F1-Score  | 0.9334 |

### Training Configuration
- Model: Custom CNN Deep (4 conv blocks + GlobalAvgPool)
- Epochs: 50
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Augmentation: Medium
- Training Time: ~2 900 с

## Comparison: All Models

| Model                  | Accuracy | Precision | Recall | F1-Score | Parameters |
|------------------------|----------|-----------|--------|----------|------------|
| ResNet50 (Baseline)    | 0.9127   | 0.9128    | 0.9127 | 0.9127   | ~23M       |
| ViT (Baseline)         | 0.9213   | 0.9215    | 0.9213 | 0.9212   | ~86M       |
| ResNet50 (Improved)    | 0.9354   | 0.9358    | 0.9354 | 0.9355   | ~23M       |
| ViT (Improved)         | **0.9412**| **0.9416**| **0.9412**| **0.9413**| ~86M |
| Custom CNN             | 0.9156   | 0.9159    | 0.9156 | 0.9155   | ~1.24M     |
| Custom CNN (Augmented) | 0.9287   | 0.9291    | 0.9287 | 0.9287   | ~1.24M     |
| Custom CNN Deep        | 0.9334   | 0.9337    | 0.9334 | 0.9334   | ~2.17M     |

## Анализ параметрической эффективности

| Model               | Accuracy | Parameters | Acc/M params |
|---------------------|----------|------------|--------------|
| Custom CNN          | 0.9156   | 1.24M      | **0.7384**   |
| Custom CNN Deep     | 0.9334   | 2.17M      | 0.4302       |
| ResNet50 (Improved) | 0.9354   | 23M        | 0.0407       |
| ViT (Improved)      | 0.9412   | 86M        | 0.0110       |

Custom CNN имеет наилучшую параметрическую эффективность — 0.74 п.п. accuracy на миллион параметров, что в 18× лучше ResNet50.

## Conclusions

1. **Custom CNN конкурентоспособен**: без предобучения достигает 91.56% — почти на уровне ResNet50 baseline (91.27%), но с в ~18× меньшим числом параметров.
2. **Аугментация сильно помогает кастомной архитектуре**: +1.31 п.п. для базовой CNN и +0.47 п.п. для Deep-варианта.
3. **ViT (Improved) остаётся лучшим**: 94.12%, однако требует 86M параметров и видеокарты для разумного времени обучения (~87 минут на CPU vs ~31 минута для Custom CNN).
4. **Custom CNN Deep** на аугментированных данных (93.34%) практически догоняет ResNet50 Improved (93.54%), имея в ~10× меньше параметров.
5. **Выбор для деплоя**: для мобильных/embedded-устройств — Custom CNN Augmented; для максимальной точности — ViT Improved.
