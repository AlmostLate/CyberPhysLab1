"""
Model evaluation module for Fashion MNIST classification.

This module provides comprehensive evaluation metrics and functions for
assessing model performance on the test set.

Metrics computed:
- Accuracy: Overall correctness
- Precision: Positive predictive value per class
- Recall: Sensitivity per class
- F1-Score: Harmonic mean of precision and recall
- Confusion Matrix: Class-wise prediction breakdown
- Classification Report: Detailed per-class metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.config import DatasetConfig, EvaluationConfig


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_labels: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate a model on the test set and compute all metrics.
    
    Args:
        model (nn.Module): PyTorch model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run evaluation on
        class_labels (Optional[List[str]]): List of class label names
    
    Returns:
        Dict[str, float]: Dictionary containing all computed metrics
    
    Example:
        >>> results = evaluate_model(model, test_loader, device)
        >>> print(f"Accuracy: {results['accuracy']:.4f}")
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    metrics = compute_metrics(
        predictions=all_predictions,
        labels=all_labels,
        average=EvaluationConfig.AVERAGE,
        class_labels=class_labels or DatasetConfig.CLASS_LABELS
    )
    
    return metrics


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    average: str = "macro",
    class_labels: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics from predictions and labels.
    
    Args:
        predictions (np.ndarray): Predicted class indices
        labels (np.ndarray): True class indices
        average (str): Averaging method for multiclass metrics
        class_labels (Optional[List[str]]): List of class label names
    
    Returns:
        Dict[str, float]: Dictionary containing all computed metrics
    
    Available metrics:
        - accuracy: Overall accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
    """
    metrics = {}
    
    # Accuracy
    metrics["accuracy"] = accuracy_score(labels, predictions)
    
    # Precision
    metrics["precision"] = precision_score(
        labels, predictions,
        average=average,
        zero_division=0
    )
    
    # Recall
    metrics["recall"] = recall_score(
        labels, predictions,
        average=average,
        zero_division=0
    )
    
    # F1 Score
    metrics["f1"] = f1_score(
        labels, predictions,
        average=average,
        zero_division=0
    )
    
    return metrics


def get_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_labels: Optional[List[str]] = None
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions (np.ndarray): Predicted class indices
        labels (np.ndarray): True class indices
        class_labels (Optional[List[str]]): List of class label names
    
    Returns:
        np.ndarray: Confusion matrix of shape (num_classes, num_classes)
    """
    return confusion_matrix(labels, predictions)


def get_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_labels: Optional[List[str]] = None
) -> str:
    """
    Generate detailed classification report.
    
    Args:
        predictions (np.ndarray): Predicted class indices
        labels (np.ndarray): True class indices
        class_labels (Optional[List[str]]): List of class label names
    
    Returns:
        str: Formatted classification report
    """
    return classification_report(
        labels, predictions,
        target_names=class_labels or DatasetConfig.CLASS_LABELS,
        digits=4
    )


def evaluate_per_class(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_labels: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Evaluate model performance per class.
    
    Args:
        model (nn.Module): PyTorch model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run evaluation on
        class_labels (Optional[List[str]]): List of class label names
    
    Returns:
        Dict[str, np.ndarray]: Dictionary with per-class metrics
    
    Example:
        >>> per_class = evaluate_per_class(model, test_loader, device)
        >>> print(f"Class 0 Precision: {per_class['precision'][0]:.4f}")
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute per-class metrics
    per_class_metrics = {
        "precision": precision_score(
            all_labels, all_predictions,
            average=None, zero_division=0
        ),
        "recall": recall_score(
            all_labels, all_predictions,
            average=None, zero_division=0
        ),
        "f1": f1_score(
            all_labels, all_predictions,
            average=None, zero_division=0
        )
    }
    
    return per_class_metrics


def print_evaluation_results(metrics: Dict[str, float]) -> None:
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metric name to value
    """
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize():15s}: {value:.4f}")
    
    print("=" * 50)


def print_per_class_results(
    per_class_metrics: Dict[str, np.ndarray],
    class_labels: Optional[List[str]] = None
) -> None:
    """
    Print per-class evaluation metrics.
    
    Args:
        per_class_metrics (Dict[str, np.ndarray]): Dictionary of per-class metrics
        class_labels (Optional[List[str]]): List of class label names
    """
    labels = class_labels or DatasetConfig.CLASS_LABELS
    
    print("=" * 70)
    print("PER-CLASS EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Class':<15s} {'Precision':>12s} {'Recall':>12s} {'F1-Score':>12s}")
    print("-" * 70)
    
    for i, label in enumerate(labels):
        print(
            f"{label:<15s} "
            f"{per_class_metrics['precision'][i]:>12.4f} "
            f"{per_class_metrics['recall'][i]:>12.4f} "
            f"{per_class_metrics['f1'][i]:>12.4f}"
        )
    
    print("=" * 70)


def compare_models(
    results: Dict[str, Dict[str, float]]
) -> None:
    """
    Compare multiple model evaluation results.
    
    Args:
        results (Dict[str, Dict[str, float]]): Dictionary mapping model names to their metrics
    
    Example:
        >>> results = {
        ...     "ResNet50": {"accuracy": 0.89, "f1": 0.88},
        ...     "ViT": {"accuracy": 0.91, "f1": 0.90}
        ... }
        >>> compare_models(results)
    """
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Print header
    metrics = list(next(iter(results.values())).keys())
    print(f"{'Model':<20s}", end="")
    for metric in metrics:
        print(f"{metric.capitalize():>15s}", end="")
    print()
    print("-" * 80)
    
    # Print each model's results
    for model_name, model_metrics in results.items():
        print(f"{model_name:<20s}", end="")
        for metric_value in model_metrics.values():
            print(f"{metric_value:>15.4f}", end="")
        print()
    
    print("=" * 80)


def save_results_to_file(
    metrics: Dict[str, float],
    filepath: str,
    model_name: str = "Model"
) -> None:
    """
    Save evaluation results to a markdown file.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics
        filepath (str): Path to save the results
        model_name (str): Name of the model being evaluated
    """
    with open(filepath, 'w') as f:
        f.write(f"# {model_name} Evaluation Results\n\n")
        f.write(f"## Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for metric_name, value in metrics.items():
            f.write(f"| {metric_name.capitalize()} | {value:.4f} |\n")
        f.write("\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Example usage
    print("Evaluation module loaded successfully")
    print("Available functions:")
    print("  - evaluate_model(): Full model evaluation")
    print("  - compute_metrics(): Compute metrics from predictions")
    print("  - get_confusion_matrix(): Get confusion matrix")
    print("  - get_classification_report(): Get detailed report")
    print("  - evaluate_per_class(): Per-class metrics")
    print("  - compare_models(): Compare multiple models")
