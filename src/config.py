"""
Configuration settings for Lab 1 CV - Street Style Classification Research.

This module contains all configurable parameters for the project including:
- Dataset settings
- Model hyperparameters
- Training configurations
- Evaluation metrics
"""

import os
from pathlib import Path


# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, CHECKPOINT_DIR, EXPERIMENTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
class DatasetConfig:
    """Configuration for Fashion MNIST dataset (Street Style Classification)."""
    
    # Dataset parameters
    NAME = "Fashion MNIST"
    NUM_CLASSES = 10
    IMAGE_SIZE = 28
    IMAGE_CHANNELS = 1
    
    # Class labels for Fashion MNIST
    CLASS_LABELS = [
        "T-shirt/top",      # 0
        "Trouser",           # 1
        "Pullover",          # 2
        "Dress",             # 3
        "Coat",              # 4
        "Sandal",            # 5
        "Shirt",             # 6
        "Sneaker",           # 7
        "Bag",               # 8
        "Ankle boot"          # 9
    ]
    
    # Data split sizes
    TRAIN_SIZE = 60000
    TEST_SIZE = 10000
    
    # Download settings
    DOWNLOAD = True
    ROOT = str(DATA_DIR)


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================
class ModelConfig:
    """Configuration for baseline models from torchvision."""
    
    # ResNet50 settings
    RESNET50 = {
        "pretrained": True,
        "num_classes": DatasetConfig.NUM_CLASSES,
        "dropout": 0.5
    }
    
    # Vision Transformer settings
    VIT = {
        "pretrained": True,
        "num_classes": DatasetConfig.NUM_CLASSES,
        "image_size": DatasetConfig.IMAGE_SIZE
    }


# =============================================================================
# TRAINING CONFIGURATIONS
# =============================================================================
class TrainingConfig:
    """Configuration for model training."""
    
    # Basic training settings
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Optimizer settings
    OPTIMIZER = "adam"
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Learning rate scheduler
    SCHEDULER = "step"
    STEP_SIZE = 10
    GAMMA = 0.1
    
    # Training duration
    EPOCHS_BASELINE = 20
    EPOCHS_IMPROVED = 30
    
    # Device settings
    DEVICE = "cuda"  # Will auto-detect if cuda is available
    SEED = 42


# =============================================================================
# AUGMENTATION CONFIGURATIONS
# =============================================================================
class AugmentationConfig:
    """Configuration for data augmentation techniques."""
    
    # Basic augmentations
    RANDOM_HORIZONTAL_FLIP = {
        "p": 0.5
    }
    
    RANDOM_ROTATION = {
        "degrees": 15
    }
    
    COLOR_JITTER = {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    }
    
    RANDOM_AFFINE = {
        "degrees": 10,
        "translate": (0.1, 0.1),
        "scale": (0.9, 1.1)
    }
    
    # Normalization (Fashion MNIST specific)
    NORMALIZE = {
        "mean": [0.2860],
        "std": [0.3530]
    }


# =============================================================================
# EVALUATION CONFIGURATIONS
# =============================================================================
class EvaluationConfig:
    """Configuration for model evaluation metrics."""
    
    # Metrics to compute
    METRICS = ["accuracy", "precision", "recall", "f1"]
    
    # Average method for multiclass metrics
    AVERAGE = "macro"  # 'micro', 'macro', 'weighted', 'samples'
    
    # Confusion matrix
    CONFUSION_MATRIX = True
    
    # Classification report
    CLASSIFICATION_REPORT = True


# =============================================================================
# IMPROVED BASELINE HYPERPARAMETERS
# =============================================================================
class ImprovedBaselineConfig:
    """Configuration for improved baseline experiments."""
    
    # Learning rate variations to test
    LEARNING_RATES = [0.01, 0.001, 0.0001]
    
    # Batch size variations
    BATCH_SIZES = [32, 64, 128]
    
    # Optimizer types
    OPTIMIZERS = ["adam", "sgd", "adamw"]
    
    # Augmentation combinations
    AUGMENTATION_LEVELS = {
        "light": ["RandomHorizontalFlip", "RandomRotation"],
        "medium": ["RandomHorizontalFlip", "RandomRotation", "ColorJitter"],
        "heavy": ["RandomHorizontalFlip", "RandomRotation", "ColorJitter", "RandomAffine"]
    }


# =============================================================================
# CUSTOM MODEL CONFIGURATION
# =============================================================================
class CustomModelConfig:
    """Configuration for custom CNN implementation."""
    
    # Architecture settings
    INPUT_CHANNELS = 1
    CONV_LAYERS = [32, 64, 128]
    KERNEL_SIZES = [3, 3, 3]
    POOL_SIZE = 2
    
    # Fully connected layers
    FC_LAYERS = [512, 256]
    DROPOUT = 0.5
    
    # Training settings for custom model
    EPOCHS = 50
    LEARNING_RATE = 0.001


# =============================================================================
# EXPERIMENT TRACKING
# =============================================================================
class ExperimentConfig:
    """Configuration for experiment tracking."""
    
    # Experiment names
    BASELINE_RESNET = "baseline_resnet50"
    BASELINE_VIT = "baseline_vit"
    IMPROVED_RESNET = "improved_resnet50"
    IMPROVED_VIT = "improved_vit"
    CUSTOM_CNN = "custom_cnn"
    
    # Logging settings
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5
    
    # Results file names
    BASELINE_RESULTS_FILE = "baseline_results.md"
    IMPROVED_RESULTS_FILE = "improved_results.md"
    CUSTOM_RESULTS_FILE = "custom_results.md"
    FINAL_REPORT_FILE = "final_report.md"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_device():
    """
    Get the appropriate device for training.
    
    Returns:
        torch.device: CUDA device if available, else CPU
    """
    import torch
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def set_seed(seed=TrainingConfig.SEED):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# CONFIGURATION INSTANCES
# =============================================================================
DATASET = DatasetConfig()
MODEL = ModelConfig()
TRAINING = TrainingConfig()
AUGMENTATION = AugmentationConfig()
EVALUATION = EvaluationConfig()
IMPROVED = ImprovedBaselineConfig()
CUSTOM = CustomModelConfig()
EXPERIMENT = ExperimentConfig()
