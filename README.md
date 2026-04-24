# Лабораторная работа 1: Computer Vision — Классификация одежды

Лабораторная работа по курсу «Киберфизические системы». Тема — исследование моделей классификации изображений, вариант на пятёрку.

## Про датасет

Используется **Fashion MNIST** — 70 000 изображений предметов одежды (28×28, оттенки серого), 10 классов: футболки, брюки, пуловеры, платья, пальто, сандалии, рубашки, кроссовки, сумки и ботинки.

Датасет выбран потому что это реальная задача e-commerce: автоматическая классификация товаров на сайте экономит ручную разметку. Классы сбалансированы, что удобно для оценки метрик. Размер 28×28 позволяет быстро гонять эксперименты даже без видеокарты.

## Метрики

- **Accuracy** — общая точность, главная метрика для сбалансированного датасета
- **Precision** — важна, чтобы не рекомендовать покупателю не тот товар
- **Recall** — важна, чтобы не пропустить подходящие категории
- **F1-Score** — сбалансированная метрика на случай дисбаланса классов

## Что сделано

### Базовые модели (torchvision)
- ResNet50 — свёрточная сеть, предобученная на ImageNet
- ViT-B/16 — трансформерная архитектура, предобученная на ImageNet

Оба вариант адаптированы под grayscale-вход 28×28 (для ViT — resize до 224×224).

### Улучшенный базлайн
Проверены гипотезы:
- Аугментации данных (горизонтальный флип, ротация, ColorJitter, RandomAffine, Cutout)
- Перебор оптимизаторов: Adam, SGD, AdamW
- Разные learning rate: 0.01, 0.001, 0.0001

### Своя реализация
Написана своя свёрточная сеть `CustomCNN` с нуля:
- 3 свёрточных блока (Conv → BN → ReLU → MaxPool)
- Dropout-регуляризация
- 2 полносвязных слоя

Также есть более глубокий вариант `CustomCNNDeep` с 4 свёрточными блоками и global average pooling.

## Структура проекта

```
Lab_24_04/
├── README.md
├── requirements.txt
├── setup.bat               # установка окружения (Windows)
├── data/                   # данные (скачиваются автоматически)
├── checkpoints/            # сохранённые чекпоинты моделей
├── src/
│   ├── config.py           # все настройки в одном месте
│   ├── dataset.py          # загрузка FashionMNIST и трансформации
│   ├── augmentations.py    # аугментации (light/medium/heavy)
│   ├── train.py            # скрипт обучения
│   ├── evaluate.py         # подсчёт метрик
│   └── models/
│       ├── baseline.py     # ResNet50 и ViT из torchvision
│       └── custom.py       # своя CNN с нуля
├── experiments/
│   ├── baseline_results.md
│   ├── improved_results.md
│   └── custom_results.md
└── reports/
    └── final_report.md
```

## Установка и запуск

### Требования
- Python 3.8+
- GPU опционально (трансформеры на CPU работают, просто дольше)

### Шаг 1 — Настройка окружения

На Windows запустить скрипт:
```bash
setup.bat
```

Или вручную:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

После этого все запуски выполнять в активированном виртуальном окружении:
```bash
venv\Scripts\activate
```

### Шаг 2 — Обучение базовых моделей

Сначала обучаем ResNet50 и ViT без аугментаций — это наш baseline для сравнения:

```bash
python src/train.py --model resnet50 --epochs 20 --experiment baseline
python src/train.py --model vit --epochs 20 --experiment baseline
```

Датасет Fashion MNIST скачается автоматически в папку `data/` при первом запуске (~30 МБ).
Чекпоинты лучших моделей сохраняются в `checkpoints/`, результаты — в `experiments/`.

### Шаг 3 — Улучшенный baseline

Добавляем аугментации и смотрим, помогает ли:

```bash
# medium: флип + ротация + ColorJitter
python src/train.py --model resnet50 --epochs 20 --augment --aug_level medium --experiment improved

# heavy: всё выше + RandomAffine
python src/train.py --model resnet50 --epochs 20 --augment --aug_level heavy --experiment improved

# можно поменять оптимизатор
python src/train.py --model resnet50 --epochs 20 --augment --optimizer adamw --experiment improved
```

### Шаг 4 — Своя реализация

Обучаем собственную CNN, написанную с нуля:

```bash
# базовый вариант (3 conv-блока)
python src/train.py --model custom --epochs 50 --augment --experiment custom

# более глубокий вариант (4 conv-блока + global average pooling)
python src/train.py --model custom_cnn_deep --epochs 50 --augment --experiment custom
```

### Шаг 5 — Оценка и сравнение

Оценить конкретный чекпоинт:
```bash
python src/evaluate.py --model resnet50 --checkpoint checkpoints/baseline_resnet50_best.pth
python src/evaluate.py --model vit --checkpoint checkpoints/baseline_vit_best.pth
python src/evaluate.py --model custom --checkpoint checkpoints/custom_custom_best.pth
```

Все результаты автоматически сохраняются в папку `experiments/` в виде markdown-файлов, которые потом идут в отчёт.

## Результаты

| Модель | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| ResNet50 (baseline) | — | — | — | — |
| ViT (baseline) | — | — | — | — |
| ResNet50 (улучшенный) | — | — | — | — |
| ViT (улучшенный) | — | — | — | — |
| CustomCNN | — | — | — | — |

Результаты заполняются после обучения.

## Стек

- PyTorch + torchvision
- scikit-learn (метрики)
- tqdm, matplotlib
