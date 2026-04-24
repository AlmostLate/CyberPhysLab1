"""
Models package for Lab 1 CV - Street Style Classification Research.

This package contains:
- baseline.py: Pre-trained models from torchvision (ResNet50, ViT)
- custom.py: Custom CNN implementation from scratch
"""

from src.models.baseline import ResNet50Model, ViTModel
from src.models.custom import CustomCNN

__all__ = ["ResNet50Model", "ViTModel", "CustomCNN"]
