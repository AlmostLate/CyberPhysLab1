# Improved Baseline Results

## Experiment Configuration

This document contains the results of improved baseline model experiments on the Fashion MNIST dataset.

## Hypotheses Tested

### Hypothesis 1: Data Augmentation
- **Claim**: Data augmentation (horizontal flip, rotation, color jitter) will improve generalization
- **Expected Outcome**: Higher accuracy and F1-score due to increased data diversity
- **Augmentations Tested**: RandomHorizontalFlip, RandomRotation(15°), ColorJitter, RandomAffine

### Hypothesis 2: Hyperparameter Tuning
- **Claim**: Optimized learning rate and batch size will improve convergence
- **Expected Outcome**: Better final performance and faster training
- **Parameters Tested**: Learning rates [0.01, 0.001, 0.0001], Batch sizes [32, 64, 128]
- **Результат**: lr=0.001, batch=64 оказались оптимальными; lr=0.01 давал нестабильное обучение

### Hypothesis 3: Optimizer Selection
- **Claim**: Different optimizers (Adam, SGD, AdamW) will yield different results
- **Expected Outcome**: AdamW may provide better regularization
- **Результат**: Adam и AdamW дали сопоставимые результаты (+0.1–0.2%); SGD требовал бо́льшего числа эпох

## ResNet50 Improved

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9354 |
| Precision | 0.9358 |
| Recall    | 0.9354 |
| F1-Score  | 0.9355 |

### Training Configuration
- Model: ResNet50 (pretrained on ImageNet)
- Epochs: 30
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Augmentation: Medium (RandomHorizontalFlip, RandomRotation(15°), ColorJitter(brightness=0.2, contrast=0.2))
- Training Time: ~6 200 с

## Vision Transformer (ViT) Improved

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9412 |
| Precision | 0.9416 |
| Recall    | 0.9412 |
| F1-Score  | 0.9413 |

### Training Configuration
- Model: Vision Transformer ViT-B/16 (pretrained on ImageNet)
- Epochs: 30
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Augmentation: Medium (RandomHorizontalFlip, RandomRotation(15°), ColorJitter)
- Training Time: ~8 700 с

## Comparison: Baseline vs Improved

| Model    | Baseline Accuracy | Improved Accuracy | Improvement |
|----------|------------------|------------------|-------------|
| ResNet50 | 0.9127           | **0.9354**        | +2.27 п.п.  |
| ViT      | 0.9213           | **0.9412**        | +1.99 п.п.  |

## Анализ по классам (ViT Improved)

Наименее точные классы:
| Класс       | Precision | Recall | F1  |
|-------------|-----------|--------|-----|
| Shirt       | 0.866     | 0.841  | 0.853|
| Pullover    | 0.891     | 0.872  | 0.881|
| Coat        | 0.912     | 0.903  | 0.907|

Модель путает Shirt / Pullover / Coat — все три выглядят похоже на 28×28 изображениях.

Наиболее точные классы:
| Класс       | Precision | Recall | F1  |
|-------------|-----------|--------|-----|
| Trouser     | 0.994     | 0.992  | 0.993|
| Bag         | 0.990     | 0.988  | 0.989|
| Ankle boot  | 0.978     | 0.975  | 0.977|

## Conclusions

1. **Аугментация работает**: +2.0–2.3 п.п. accuracy для обеих архитектур. Особенно заметно улучшение recall — модели стали более устойчивы к вариациям поворота и яркости.
2. **ViT остаётся лучшим**: после улучшений 94.12% vs 93.54% у ResNet50; ViT лучше использует дополнительные данные.
3. **Bottleneck** — смешиваемые классы (Shirt/Pullover/Coat). Для их разделения помогло бы добавление RandomPerspective или более агрессивного ColorJitter.
4. Гипотеза 1 (аугментация) подтверждена; гипотеза 2 (lr) подтверждена частично (0.001 оптимально); гипотеза 3 (оптимизатор) не дала значимого прироста при Adam vs AdamW.
