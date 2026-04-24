"""
Data augmentation transforms for Fashion MNIST classification.

This module provides various data augmentation techniques to improve
model generalization and prevent overfitting. Augmentations are applied
during training only.

Available augmentations:
- RandomHorizontalFlip: Random horizontal flip
- RandomRotation: Random rotation within specified degrees
- ColorJitter: Random color/brightness adjustments
- RandomAffine: Random affine transformations
- RandomErasing: Random erasing for regularization
- MixUp: MixUp augmentation for better generalization
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Optional, Tuple

from src.config import AugmentationConfig, DatasetConfig


class RandomErasing(transforms.RandomErasing):
    """
    Random erasing augmentation for Fashion MNIST.
    
    Randomly selects a rectangle region in the image and erases its pixels
    with random values. This helps the model become robust to occlusions.
    
    Example:
        >>> transform = transforms.Compose([
        ...     transforms.ToTensor(),
        ...     RandomErasing(p=0.5),
        ...     transforms.Normalize(mean=[0.2860], std=[0.3530])
        ... ])
    """
    pass


class Cutout(nn.Module):
    """
    Cutout augmentation - randomly mask out square regions.
    
    This is a simpler alternative to RandomErasing that masks out
    square regions with a constant value (usually the mean).
    
    Attributes:
        n_holes (int): Number of holes to cut out
        length (int): Length of the square hole
    
    Example:
        >>> cutout = Cutout(n_holes=1, length=8)
        >>> img = cutout(img)
    """
    
    def __init__(self, n_holes: int = 1, length: int = 8):
        """
        Initialize Cutout augmentation.
        
        Args:
            n_holes (int): Number of holes to cut out. Default: 1
            length (int): Length of the square hole. Default: 8
        """
        super(Cutout, self).__init__()
        self.n_holes = n_holes
        self.length = length
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply cutout to the image.
        
        Args:
            img (torch.Tensor): Image tensor of shape (C, H, W)
        
        Returns:
            torch.Tensor: Image with cutout applied
        """
        h = img.size(1)
        w = img.size(2)
        
        mask = torch.ones((h, w), dtype=torch.float32)
        
        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            
            mask[y1:y2, x1:x2] = 0.0
        
        mask = mask.expand_as(img)
        img = img * mask
        
        return img


class MixUp:
    """
    MixUp augmentation for batch-level data augmentation.
    
    MixUp creates virtual training examples by mixing pairs of images
    and their labels. This improves model generalization and reduces
    memorization of training labels.
    
    Reference: https://arxiv.org/abs/1710.09412
    
    Attributes:
        alpha (float): MixUp beta distribution parameter
    
    Example:
        >>> mixup = MixUp(alpha=1.0)
        >>> mixed_x, mixed_y = mixup(images, labels)
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize MixUp augmentation.
        
        Args:
            alpha (float): Parameter for beta distribution. Default: 1.0
        """
        self.alpha = alpha
    
    def __call__(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply MixUp to a batch.
        
        Args:
            batch_x (torch.Tensor): Batch of images
            batch_y (torch.Tensor): Batch of labels (class indices)
        
        Returns:
            Tuple containing:
                - Mixed images
                - Mixed labels (for soft labels)
                - Original labels
                - Lambda value used for mixing
        """
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1.0
        
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        y_a, y_b = batch_y, batch_y[index]
        
        return mixed_x, y_a, y_b, lam


def get_light_augmentation() -> transforms.Compose:
    """
    Get light augmentation pipeline (minimal transformations).
    
    Light augmentations are suitable for initial experiments or when
    the dataset is already diverse.
    
    Returns:
        transforms.Compose: Light augmentation pipeline
    
    Augmentations:
        - RandomHorizontalFlip (p=0.5)
        - RandomRotation (degrees=15)
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(**AugmentationConfig.RANDOM_HORIZONTAL_FLIP),
        transforms.RandomRotation(**AugmentationConfig.RANDOM_ROTATION),
    ])


def get_medium_augmentation() -> transforms.Compose:
    """
    Get medium augmentation pipeline (balanced transformations).
    
    Medium augmentations provide a good balance between diversity
    and preserving original image characteristics.
    
    Returns:
        transforms.Compose: Medium augmentation pipeline
    
    Augmentations:
        - RandomHorizontalFlip (p=0.5)
        - RandomRotation (degrees=15)
        - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(**AugmentationConfig.RANDOM_HORIZONTAL_FLIP),
        transforms.RandomRotation(**AugmentationConfig.RANDOM_ROTATION),
        transforms.ColorJitter(**AugmentationConfig.COLOR_JITTER),
    ])


def get_heavy_augmentation() -> transforms.Compose:
    """
    Get heavy augmentation pipeline (aggressive transformations).
    
    Heavy augmentations are suitable when more data diversity is needed
    or when training for longer periods with potential overfitting.
    
    Returns:
        transforms.Compose: Heavy augmentation pipeline
    
    Augmentations:
        - RandomHorizontalFlip (p=0.5)
        - RandomRotation (degrees=15)
        - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        - RandomAffine (degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(**AugmentationConfig.RANDOM_HORIZONTAL_FLIP),
        transforms.RandomRotation(**AugmentationConfig.RANDOM_ROTATION),
        transforms.ColorJitter(**AugmentationConfig.COLOR_JITTER),
        transforms.RandomAffine(**AugmentationConfig.RANDOM_AFFINE),
    ])


def get_augmentation_by_level(level: str) -> transforms.Compose:
    """
    Get augmentation pipeline by level name.
    
    Args:
        level (str): Augmentation level ('light', 'medium', or 'heavy')
    
    Returns:
        transforms.Compose: Corresponding augmentation pipeline
    
    Raises:
        ValueError: If level is not recognized
    
    Example:
        >>> transform = get_augmentation_by_level('medium')
    """
    level = level.lower()
    
    if level == "light":
        return get_light_augmentation()
    elif level == "medium":
        return get_medium_augmentation()
    elif level == "heavy":
        return get_heavy_augmentation()
    else:
        raise ValueError(
            f"Unknown augmentation level: {level}. "
            f"Available levels: 'light', 'medium', 'heavy'"
        )


def get_augmentation_with_cutout(level: str = "medium", cutout_length: int = 8) -> transforms.Compose:
    """
    Get augmentation pipeline with Cutout added.
    
    Args:
        level (str): Base augmentation level ('light', 'medium', 'heavy')
        cutout_length (int): Length of cutout square. Default: 8
    
    Returns:
        transforms.Compose: Augmentation pipeline with Cutout
    
    Example:
        >>> transform = get_augmentation_with_cutout('medium', cutout_length=8)
    """
    base_augmentation = get_augmentation_by_level(level)
    
    return transforms.Compose([
        base_augmentation,
        Cutout(n_holes=1, length=cutout_length),
    ])


def visualize_augmentations(
    image: torch.Tensor,
    augmentations: transforms.Compose,
    n_samples: int = 5
) -> torch.Tensor:
    """
    Visualize the effect of augmentations on an image.
    
    Args:
        image (torch.Tensor): Input image tensor
        augmentations (transforms.Compose): Augmentation pipeline
        n_samples (int): Number of augmented samples to generate
    
    Returns:
        torch.Tensor: Grid of augmented images
    
    Example:
        >>> augmented = visualize_augmentations(img, get_medium_augmentation(), 5)
    """
    augmented_images = []
    
    for _ in range(n_samples):
        augmented = augmentations(image)
        augmented_images.append(augmented)
    
    return torch.stack(augmented_images)


# =============================================================================
# AUGMENTATION HYPERPARAMETERS FOR TUNING
# =============================================================================
class AugmentationHyperparameters:
    """
    Hyperparameters for augmentation techniques.
    
    These can be tuned to find optimal augmentation strategies.
    """
    
    # Rotation
    ROTATION_DEGREES_OPTIONS = [10, 15, 20, 25]
    
    # Horizontal Flip
    FLIP_PROB_OPTIONS = [0.3, 0.5, 0.7]
    
    # Color Jitter
    BRIGHTNESS_OPTIONS = [0.1, 0.2, 0.3]
    CONTRAST_OPTIONS = [0.1, 0.2, 0.3]
    SATURATION_OPTIONS = [0.1, 0.2, 0.3]
    HUE_OPTIONS = [0.05, 0.1, 0.15]
    
    # Affine
    TRANSLATE_OPTIONS = [(0.05, 0.05), (0.1, 0.1), (0.15, 0.15)]
    SCALE_OPTIONS = [(0.9, 1.1), (0.85, 1.15), (0.8, 1.2)]
    
    # Cutout
    CUTOUT_LENGTH_OPTIONS = [4, 8, 12, 16]
    
    # MixUp
    MIXUP_ALPHA_OPTIONS = [0.2, 0.5, 1.0, 2.0]


def get_default_augmentation_config() -> dict:
    """
    Get default augmentation configuration.
    
    Returns:
        dict: Dictionary with default augmentation settings
    """
    return {
        "rotation_degrees": 15,
        "flip_prob": 0.5,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "translate": (0.1, 0.1),
        "scale": (0.9, 1.1),
        "cutout_length": 8,
        "mixup_alpha": 1.0
    }
