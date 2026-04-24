"""
Dataset loading and preprocessing for Fashion MNIST (Street Style Classification).

This module provides utilities for:
- Loading Fashion MNIST dataset
- Applying transformations and augmentations
- Creating data loaders

Fashion MNIST is used as a proxy for street style detection, representing
10 categories of clothing items commonly found in street fashion photography.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Optional

from src.config import (
    DatasetConfig,
    TrainingConfig,
    AugmentationConfig,
    get_device
)


def get_transforms(train: bool = True, augment: bool = False) -> transforms.Compose:
    """
    Get data transformations for Fashion MNIST dataset.
    
    Args:
        train (bool): If True, return training transforms (with optional augmentation).
                      If False, return test/validation transforms.
        augment (bool): If True, apply data augmentation for training.
    
    Returns:
        transforms.Compose: Composed transformation pipeline
    
    Example:
        >>> train_transform = get_transforms(train=True, augment=True)
        >>> test_transform = get_transforms(train=False)
    """
    if train:
        transform_list = [
            transforms.Resize((DatasetConfig.IMAGE_SIZE, DatasetConfig.IMAGE_SIZE)),
        ]
        
        if augment:
            # Add augmentation techniques
            transform_list.extend([
                transforms.RandomHorizontalFlip(**AugmentationConfig.RANDOM_HORIZONTAL_FLIP),
                transforms.RandomRotation(**AugmentationConfig.RANDOM_ROTATION),
                transforms.ColorJitter(**AugmentationConfig.COLOR_JITTER),
                transforms.RandomAffine(**AugmentationConfig.RANDOM_AFFINE),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(**AugmentationConfig.NORMALIZE)
        ])
    else:
        transform_list = [
            transforms.Resize((DatasetConfig.IMAGE_SIZE, DatasetConfig.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(**AugmentationConfig.NORMALIZE)
        ]
    
    return transforms.Compose(transform_list)


def load_fashion_mnist(
    train: bool = True,
    augment: bool = False,
    download: bool = True
) -> datasets.FashionMNIST:
    """
    Load Fashion MNIST dataset with appropriate transformations.
    
    Args:
        train (bool): If True, load training set. If False, load test set.
        augment (bool): If True, apply data augmentation to training set.
        download (bool): If True, download dataset if not present.
    
    Returns:
        FashionMNIST: Fashion MNIST dataset with applied transformations
    
    Example:
        >>> train_dataset = load_fashion_mnist(train=True, augment=True)
        >>> test_dataset = load_fashion_mnist(train=False)
    """
    transform = get_transforms(train=train, augment=augment)
    
    dataset = datasets.FashionMNIST(
        root=DatasetConfig.ROOT,
        train=train,
        transform=transform,
        download=download
    )
    
    return dataset


def create_data_loaders(
    batch_size: int = TrainingConfig.BATCH_SIZE,
    augment: bool = False,
    val_split: float = 0.1,
    random_seed: int = TrainingConfig.SEED
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        batch_size (int): Number of samples per batch.
        augment (bool): If True, apply data augmentation to training set.
        val_split (float): Fraction of training data to use for validation.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: 
            Train, validation, and test data loaders
    
    Example:
        >>> train_loader, val_loader, test_loader = create_data_loaders(
        ...     batch_size=64, augment=True
        ... )
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Load full training dataset
    full_train_dataset = load_fashion_mnist(train=True, augment=False, download=True)
    test_dataset = load_fashion_mnist(train=False, augment=False, download=True)
    
    # Split training data into train and validation
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=TrainingConfig.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=TrainingConfig.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=TrainingConfig.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader


def get_class_distribution(dataset: datasets.FashionMNIST) -> dict:
    """
    Get the class distribution in the dataset.
    
    Args:
        dataset: Fashion MNIST dataset
    
    Returns:
        dict: Dictionary mapping class indices to sample counts
    """
    class_counts = {}
    
    for _, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    return class_counts


def print_dataset_info(train_loader: DataLoader, val_loader: DataLoader, 
                       test_loader: DataLoader) -> None:
    """
    Print information about the dataset loaders.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    """
    print("=" * 50)
    print("Dataset Information")
    print("=" * 50)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {DatasetConfig.NUM_CLASSES}")
    print(f"Class labels: {DatasetConfig.CLASS_LABELS}")
    print(f"Image size: {DatasetConfig.IMAGE_SIZE}x{DatasetConfig.IMAGE_SIZE}")
    print(f"Batch size: {train_loader.batch_size}")
    print("=" * 50)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Test data loading
    print("Testing dataset loading...")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=64,
        augment=True
    )
    
    print_dataset_info(train_loader, val_loader, test_loader)
    
    # Print class distribution
    print("\nClass distribution in training set:")
    full_train_dataset = load_fashion_mnist(train=True, augment=False)
    class_dist = get_class_distribution(full_train_dataset)
    
    for class_idx, count in sorted(class_dist.items()):
        label = DatasetConfig.CLASS_LABELS[class_idx]
        percentage = (count / len(full_train_dataset)) * 100
        print(f"  {class_idx}: {label:15s} - {count:5d} samples ({percentage:.1f}%)")
