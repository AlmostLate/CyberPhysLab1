"""
Custom CNN implementation from scratch for Fashion MNIST classification.

This module provides a custom convolutional neural network architecture
implemented without using pre-trained weights. The architecture is designed
specifically for 28x28 grayscale fashion images with 10 output classes.

Architecture:
- 3 convolutional blocks with batch normalization and max pooling
- Dropout for regularization
- 2 fully connected layers for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from src.config import DatasetConfig, CustomModelConfig


class CustomCNN(nn.Module):
    """
    Custom Convolutional Neural Network for Fashion MNIST classification.
    
    This CNN is designed from scratch without pre-trained weights. It consists of:
    - 3 convolutional blocks (Conv -> BatchNorm -> ReLU -> MaxPool)
    - Dropout layers for regularization
    - 2 fully connected layers for final classification
    
    Architecture:
        Conv Block 1: 1 -> 32 channels, 3x3 kernel
        Conv Block 2: 32 -> 64 channels, 3x3 kernel
        Conv Block 3: 64 -> 128 channels, 3x3 kernel
        FC Layers: 128*3*3 -> 512 -> 256 -> 10
    
    Attributes:
        conv1, conv2, conv3: Convolutional layers
        bn1, bn2, bn3: Batch normalization layers
        fc1, fc2: Fully connected layers
        dropout: Dropout layer
    
    Example:
        >>> model = CustomCNN(num_classes=10)
        >>> output = model(torch.randn(1, 1, 28, 28))
        >>> print(output.shape)  # torch.Size([1, 10])
    """
    
    def __init__(
        self,
        num_classes: int = DatasetConfig.NUM_CLASSES,
        input_channels: int = CustomModelConfig.INPUT_CHANNELS,
        conv_channels: List[int] = CustomModelConfig.CONV_LAYERS,
        kernel_sizes: List[int] = CustomModelConfig.KERNEL_SIZES,
        pool_size: int = CustomModelConfig.POOL_SIZE,
        fc_layers: List[int] = CustomModelConfig.FC_LAYERS,
        dropout: float = CustomModelConfig.DROPOUT
    ):
        """
        Initialize Custom CNN model.
        
        Args:
            num_classes (int): Number of output classes. Default: 10
            input_channels (int): Number of input channels. Default: 1 (grayscale)
            conv_channels (List[int]): Number of channels for each conv layer.
            kernel_sizes (List[int]): Kernel sizes for each conv layer.
            pool_size (int): Max pooling kernel size.
            fc_layers (List[int]): Hidden layer sizes for fully connected layers.
            dropout (float): Dropout probability.
        """
        super(CustomCNN, self).__init__()
        
        # Convolutional Block 1: 1x28x28 -> 32x14x14
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv_channels[0],
            kernel_size=kernel_sizes[0],
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        
        # Convolutional Block 2: 32x14x14 -> 64x7x7
        self.conv2 = nn.Conv2d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=kernel_sizes[1],
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        
        # Convolutional Block 3: 64x7x7 -> 128x3x3
        self.conv3 = nn.Conv2d(
            in_channels=conv_channels[1],
            out_channels=conv_channels[2],
            kernel_size=kernel_sizes[2],
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(conv_channels[2])
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=2)
        
        # Dropout
        self.dropout = nn.Dropout2d(p=dropout)
        
        # Calculate the size of features after conv layers
        # Input: 28x28 -> after conv1+pool: 14x14 -> after conv2+pool: 7x7 -> after conv3+pool: 3x3
        feature_size = conv_channels[2] * 3 * 3
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(feature_size, fc_layers[0])
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(fc_layers[0], fc_layers[1])
        
        # Output Layer
        self.fc3 = nn.Linear(fc_layers[1], num_classes)
        
        # Store model name
        self.model_name = "CustomCNN"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Conv Block 1: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # FC Layer 1 with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # FC Layer 2 with dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classification layer.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Feature tensor before final classification layer
        """
        # Conv blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers (without final output)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x


class CustomCNNDeep(nn.Module):
    """
    Deeper variant of Custom CNN with more layers and regularization.
    
    This is an alternative architecture with:
    - 4 convolutional blocks
    - Global average pooling
    - More aggressive dropout
    
    Use this variant for potentially better performance on complex patterns.
    """
    
    def __init__(
        self,
        num_classes: int = DatasetConfig.NUM_CLASSES,
        dropout: float = 0.5
    ):
        """
        Initialize deeper Custom CNN model.
        
        Args:
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability.
        """
        super(CustomCNNDeep, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Conv Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.model_name = "CustomCNNDeep"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the deeper network.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output logits
        """
        # Conv blocks with residual-like structure
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


def get_custom_model(
    model_name: str = "custom_cnn",
    num_classes: int = DatasetConfig.NUM_CLASSES,
    dropout: float = CustomModelConfig.DROPOUT
) -> nn.Module:
    """
    Factory function to get custom model by name.
    
    Args:
        model_name (str): Name of the model ('custom_cnn' or 'custom_cnn_deep')
        num_classes (int): Number of output classes
        dropout (float): Dropout probability
    
    Returns:
        nn.Module: Initialized custom model
    
    Raises:
        ValueError: If model_name is not recognized
    
    Example:
        >>> model = get_custom_model('custom_cnn', num_classes=10)
        >>> model = get_custom_model('custom_cnn_deep', num_classes=10)
    """
    model_name = model_name.lower()
    
    if model_name == "custom_cnn":
        return CustomCNN(
            num_classes=num_classes,
            dropout=dropout
        )
    elif model_name == "custom_cnn_deep":
        return CustomCNNDeep(
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Available models: 'custom_cnn', 'custom_cnn_deep'"
        )


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module) -> None:
    """
    Print information about the custom model architecture.
    
    Args:
        model (nn.Module): PyTorch model
    """
    print("=" * 50)
    print(f"Model: {model.model_name if hasattr(model, 'model_name') else 'Unknown'}")
    print("=" * 50)
    print(f"Total parameters: {count_parameters(model):,}")
    print("=" * 50)
