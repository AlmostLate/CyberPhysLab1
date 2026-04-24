"""
Training pipeline for Fashion MNIST classification models.

This module provides a complete training pipeline including:
- Baseline model training (ResNet50, ViT from torchvision)
- Improved baseline with data augmentation and hyperparameter tuning
- Custom CNN training
- Checkpoint saving and loading
- Training progress visualization

Usage:
    # Train ResNet50 baseline
    python src/train.py --model resnet50 --epochs 20
    
    # Train with augmentation
    python src/train.py --model resnet50 --epochs 20 --augment
    
    # Train custom model
    python src/train.py --model custom --epochs 50 --augment
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DatasetConfig,
    TrainingConfig,
    ExperimentConfig,
    CHECKPOINT_DIR,
    EXPERIMENTS_DIR,
    get_device,
    set_seed
)
from src.dataset import create_data_loaders, print_dataset_info
from src.models.baseline import get_baseline_model, count_parameters as count_baseline_params
from src.models.custom import get_custom_model, count_parameters as count_custom_params
from src.augmentations import get_augmentation_by_level
from src.evaluate import evaluate_model, print_evaluation_results


def parse_args():
    """
    Parse command line arguments for training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train Fashion MNIST classification models"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "vit", "custom", "custom_cnn_deep"],
        help="Model to train (resnet50, vit, custom)"
    )
    
    # Training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    # Augmentation settings
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation"
    )
    parser.add_argument(
        "--aug_level",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Augmentation level"
    )
    
    # Optimizer settings
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "adamw"],
        help="Optimizer type"
    )
    
    # Experiment settings
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        choices=["baseline", "improved", "custom"],
        help="Experiment type"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Custom experiment name"
    )
    
    # Checkpoint settings
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training"
    )
    parser.add_argument(
        "--save_checkpoint",
        action="store_true",
        default=True,
        help="Save checkpoints during training"
    )
    
    # Other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed training progress"
    )
    
    return parser.parse_args()


def get_model(model_name: str, device: torch.device) -> nn.Module:
    """
    Initialize model based on name.
    
    Args:
        model_name (str): Name of the model
        device (torch.device): Device to load model on
    
    Returns:
        nn.Module: Initialized model
    """
    model_name = model_name.lower()
    
    if model_name == "resnet50":
        model = get_baseline_model("resnet50", pretrained=True)
    elif model_name == "vit":
        model = get_baseline_model("vit", pretrained=True)
    elif model_name in ["custom", "custom_cnn"]:
        model = get_custom_model("custom_cnn")
    elif model_name == "custom_cnn_deep":
        model = get_custom_model("custom_cnn_deep")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float
) -> optim.Optimizer:
    """
    Get optimizer for training.
    
    Args:
        model (nn.Module): Model to optimize
        optimizer_name (str): Name of optimizer
        learning_rate (float): Learning rate
    
    Returns:
        optim.Optimizer: Configured optimizer
    """
    if optimizer_name.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=TrainingConfig.WEIGHT_DECAY
        )
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=TrainingConfig.WEIGHT_DECAY
        )
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=TrainingConfig.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: optim.Optimizer,
    step_size: int = TrainingConfig.STEP_SIZE
) -> optim.lr_scheduler:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer (optim.Optimizer): Optimizer to schedule
        step_size (int): Step size for StepLR scheduler
    
    Returns:
        optim.lr_scheduler: Learning rate scheduler
    """
    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=TrainingConfig.GAMMA
    )


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    verbose: bool = False
) -> float:
    """
    Train for one epoch.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        epoch (int): Current epoch number
        verbose (bool): Whether to print progress
    
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if verbose else train_loader
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if verbose:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.2f}%"
            })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to validate on
    
    Returns:
        Tuple[float, float]: Validation loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    experiment_name: str,
    save_checkpoint: bool = True,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Complete training pipeline.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        optimizer (optim.Optimizer): Optimizer
        scheduler (optim.lr_scheduler): Learning rate scheduler
        criterion (nn.Module): Loss function
        device (torch.device): Device to train on
        epochs (int): Number of epochs
        experiment_name (str): Name for this experiment
        save_checkpoint (bool): Whether to save checkpoints
        verbose (bool): Whether to print detailed progress
    
    Returns:
        Dict[str, float]: Final evaluation metrics
    """
    best_val_acc = 0.0
    best_model_state = None
    
    print("=" * 60)
    print(f"Starting training: {experiment_name}")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, verbose
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            
            if save_checkpoint:
                checkpoint_path = CHECKPOINT_DIR / f"{experiment_name}_best.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, checkpoint_path)
                print(f"  -> Saved best model to {checkpoint_path}")
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    metrics = evaluate_model(model, test_loader, device)
    print_evaluation_results(metrics)
    
    return metrics


def main():
    """
    Main training function.
    """
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=args.batch_size,
        augment=args.augment
    )
    print_dataset_info(train_loader, val_loader, test_loader)
    
    # Initialize model
    print(f"\nInitializing model: {args.model}")
    model = get_model(args.model, device)
    
    # Count parameters
    if args.model.lower() in ["resnet50", "vit"]:
        param_count = count_baseline_params(model)
    else:
        param_count = count_custom_params(model)
    print(f"Trainable parameters: {param_count:,}")
    
    # Get optimizer and scheduler
    optimizer = get_optimizer(model, args.optimizer, args.lr)
    scheduler = get_scheduler(optimizer)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Experiment name
    experiment_name = args.experiment_name or f"{args.experiment}_{args.model}"
    if args.augment:
        experiment_name += f"_aug_{args.aug_level}"
    
    # Train model
    start_time = time.time()
    
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        experiment_name=experiment_name,
        save_checkpoint=args.save_checkpoint,
        verbose=args.verbose
    )
    
    training_time = time.time() - start_time
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Test F1-Score: {metrics['f1']:.4f}")
    print("=" * 60)
    
    # Save results to file
    results_file = EXPERIMENTS_DIR / f"{experiment_name}_results.md"
    with open(results_file, 'w') as f:
        f.write(f"# {experiment_name} Results\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Model: {args.model}\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Batch Size: {args.batch_size}\n")
        f.write(f"- Learning Rate: {args.lr}\n")
        f.write(f"- Optimizer: {args.optimizer}\n")
        f.write(f"- Augmentation: {args.augment} ({args.aug_level})\n")
        f.write(f"- Training Time: {training_time:.2f}s\n\n")
        f.write(f"## Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for metric_name, value in metrics.items():
            f.write(f"| {metric_name.capitalize()} | {value:.4f} |\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
