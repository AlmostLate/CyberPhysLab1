# Baseline Model Results

## Experiment Configuration

This document contains the results of baseline model experiments on the Fashion MNIST dataset.

## ResNet50 Baseline

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9127 |
| Precision | 0.9128 |
| Recall    | 0.9127 |
| F1-Score  | 0.9127 |

### Training Configuration
- Model: ResNet50 (pretrained on ImageNet)
- Epochs: 20
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Augmentation: None
- Training Time: 4189.41 с (~70 минут)

### Примечания

Модель обучалась с замороженными слоями-экстрактором (только полносвязный классификационный слой дообучался). Fashion MNIST изображения 28×28 апсемплируются до 224×224 для совместимости с ResNet50. Результат 91.27% является сильным стартом для transfer learning.

## Vision Transformer (ViT) Baseline

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9213 |
| Precision | 0.9215 |
| Recall    | 0.9213 |
| F1-Score  | 0.9212 |

### Training Configuration
- Model: Vision Transformer ViT-B/16 (pretrained on ImageNet)
- Epochs: 20
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Augmentation: None
- Training Time: ~5 800 с (~97 минут)

### Примечания

ViT-B/16 делит изображение на патчи 16×16 — при 224×224 получается 196 патчей. Несмотря на значительно большее число параметров (~86M vs ~23M у ResNet50), ViT превышает ResNet50 на ~0.86 п.п. благодаря механизму self-attention, который эффективнее захватывает глобальные паттерны текстуры ткани.

## Comparison

| Model   | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| ResNet50 | 0.9127  | 0.9128    | 0.9127 | 0.9127   |
| ViT      | **0.9213**| **0.9215**| **0.9213**| **0.9212** |

## Conclusions

- Оба предобученных на ImageNet базовых классификатора показывают точность >91% без каких-либо аугментаций.
- ViT незначительно превосходит ResNet50 (+0.86 п.п.), что согласуется с литературными данными по Fashion MNIST.
- Baseline-результаты служат точкой отсчёта для экспериментов с аугментациями (improved baseline) и кастомной CNN.
