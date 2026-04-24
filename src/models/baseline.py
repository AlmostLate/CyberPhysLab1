"""
Baseline models from torchvision for Fashion MNIST classification.

This module provides wrapper classes for pre-trained models from torchvision:
- ResNet50: Convolutional neural network baseline
- Vision Transformer (ViT): Transformer-based baseline

These models serve as baselines for comparison with custom implementations.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

from src.config import DatasetConfig, ModelConfig


class ResNet50Model(nn.Module):
    """
    ResNet50 model adapted for Fashion MNIST classification.
    
    ResNet50 is a convolutional neural network with 50 layers, pre-trained
    on ImageNet. This wrapper modifies it for 28x28 grayscale input with
    10 output classes.
    
    Attributes:
        model (nn.Module): The underlying ResNet50 model
    
    Example:
        >>> model = ResNet50Model(num_classes=10, pretrained=True)
        >>> output = model(torch.randn(1, 1, 28, 28))
        >>> print(output.shape)  # torch.Size([1, 10])
    """
    
    def __init__(
        self,
        num_classes: int = DatasetConfig.NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Initialize ResNet50 model.
        
        Args:
            num_classes (int): Number of output classes. Default: 10
            pretrained (bool): Whether to load ImageNet pretrained weights.
            dropout (float): Dropout probability for regularization.
        """
        super(ResNet50Model, self).__init__()
        
        # Load pre-trained ResNet50 (using weights API, compatible with torchvision >= 0.13)
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        
        # Modify first conv layer for grayscale input (1 channel instead of 3)
        # Fashion MNIST images are 28x28, ResNet expects 224x224
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Remove avgpool and modify fc layer for our number of classes
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        # Store model name for reference
        self.model_name = "ResNet50"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classification layer.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Feature tensor before classification layer
        """
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)
        
        features = self.model.layer1(features)
        features = self.model.layer2(features)
        features = self.model.layer3(features)
        features = self.model.layer4(features)
        
        features = self.model.avgpool(features)
        features = torch.flatten(features, 1)
        
        return features


class ViTModel(nn.Module):
    """
    Vision Transformer (ViT) model adapted for Fashion MNIST classification.
    
    ViT is a transformer-based architecture that processes images as sequences
    of patches. This wrapper modifies it for 28x28 grayscale input with 10
    output classes.
    
    Attributes:
        model (nn.Module): The underlying ViT model
    
    Example:
        >>> model = ViTModel(num_classes=10, pretrained=True)
        >>> output = model(torch.randn(1, 1, 28, 28))
        >>> print(output.shape)  # torch.Size([1, 10])
    """
    
    def __init__(
        self,
        num_classes: int = DatasetConfig.NUM_CLASSES,
        pretrained: bool = True,
        image_size: int = DatasetConfig.IMAGE_SIZE
    ):
        """
        Initialize Vision Transformer model.
        
        Args:
            num_classes (int): Number of output classes. Default: 10
            pretrained (bool): Whether to load ImageNet pretrained weights.
            image_size (int): Input image size. Default: 28
        """
        super(ViTModel, self).__init__()
        
        # Load pre-trained ViT-Base model (using weights API, compatible with torchvision >= 0.13)
        # ViT requires 224x224 input, so we need to handle smaller images
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = models.vit_b_16(weights=weights)
        
        # Get the original classifier input dimension
        in_features = self.model.heads.head.in_features
        
        # Replace the classification head
        self.model.heads.head = nn.Linear(in_features, num_classes)
        
        # Store image size for preprocessing
        self.image_size = image_size
        self.model_name = "VisionTransformer"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # ViT expects 3-channel input, convert grayscale to RGB by repeating
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # ViT expects 224x224, resize if needed
        if x.size(2) != 224 or x.size(3) != 224:
            x = nn.functional.interpolate(
                x,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
        
        return self.model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classification layer.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Feature tensor before classification layer
        """
        # Convert to RGB and resize
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        if x.size(2) != 224 or x.size(3) != 224:
            x = nn.functional.interpolate(
                x,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
        
        # Get features from the encoder (before classification head)
        x = self.model._process_input(x)
        batch_size = x.size(0)
        
        # Create patch embeddings
        x = self.model.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_token = self.model.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional embeddings
        x = x + self.model.pos_embed
        
        # Pass through encoder
        x = self.model.encoder(x)
        
        # Get CLS token output
        features = x[:, 0]
        
        return features


def get_baseline_model(
    model_name: str,
    num_classes: int = DatasetConfig.NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.5
) -> nn.Module:
    """
    Factory function to get baseline model by name.
    
    Args:
        model_name (str): Name of the model ('resnet50' or 'vit')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout probability
    
    Returns:
        nn.Module: Initialized model
    
    Raises:
        ValueError: If model_name is not recognized
    
    Example:
        >>> model = get_baseline_model('resnet50', num_classes=10)
        >>> model = get_baseline_model('vit', num_classes=10)
    """
    model_name = model_name.lower()
    
    if model_name == "resnet50":
        return ResNet50Model(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_name in ["vit", "vision_transformer", "transformer"]:
        return ViTModel(
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=DatasetConfig.IMAGE_SIZE
        )
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Available models: 'resnet50', 'vit'"
        )


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Number of trainable parameters
    
    Example:
        >>> model = ResNet50Model()
        >>> param_count = count_parameters(model)
        >>> print(f"Trainable parameters: {param_count:,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module) -> None:
    """
    Print information about the model architecture.
    
    Args:
        model (nn.Module): PyTorch model
    """
    print("=" * 50)
    print(f"Model: {model.model_name if hasattr(model, 'model_name') else 'Unknown'}")
    print("=" * 50)
    print(f"Total parameters: {count_parameters(model):,}")
    print("=" * 50)
