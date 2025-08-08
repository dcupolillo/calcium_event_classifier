""" Created on July 28, 2025
    @author: dcupolillo """

from pathlib import Path
import calcium_event_classifier as cec
import calcium_event_classifier.core.plot as cec_plot
import flammkuchen as fl
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score)

device = cec.set_device()

# Load model
model_path = Path(
    r"models/your_model_with_dff.pth")

checkpoint = torch.load(model_path, map_location=device)

# Print available keys in checkpoint
print(f"Checkpoint keys: {list(checkpoint.keys())}")

# Initialize MinimalClassifier with saved hyperparameters
hyperparams = checkpoint['hyperparams']
print("\nModel hyperparameters:")
for key, value in hyperparams.items():
    print(f"  {key}: {value}")

# Initialize MinimalClassifier
classifier = cec.CalciumEventClassifier2Ch(
    conv1_channels=hyperparams["conv1_channels"],
    conv2_channels=hyperparams["conv2_channels"],
    conv3_channels=hyperparams["conv3_channels"],
    conv1_kernel=hyperparams["conv1_kernel"],
    conv2_kernel=hyperparams["conv2_kernel"],
    conv3_kernel=hyperparams["conv3_kernel"],
    leaky_relu_negative_slope=hyperparams["leaky_relu_negative_slope"],
    dropout_rate=hyperparams["dropout_rate"],
    pool_kernel=hyperparams["pool_kernel"]
).to(device)

# Load model weights
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()

print("\nModel architecture:")
print(classifier)

# Print model parameters summary
total_params = sum(p.numel() for p in classifier.parameters())
trainable_params = sum(
    p.numel() for p in classifier.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Optional: Print parameter statistics
print("\nParameter statistics:")
for name, param in classifier.named_parameters():
    if param.requires_grad:
        print(f"  {name} | shape: {param.shape} | "
              f"mean: {param.data.mean():.4f}, "
              f"std: {param.data.std():.4f}")

# Initialize test dataset
test_data_path = Path(
    r"datasets/test_dataset_250530_ratio_3_1.h5")
test_data = fl.load(test_data_path)

test_dataset = cec.ZScoreDataset(
    test_data,
    augment=False,
)

test_loader = cec.load_test_dataset(
    test_dataset,
    batch_size=checkpoint['hyperparams']["batch_size"],
    summary=False
)

labels, logits = cec.get_predictions_and_labels(
    classifier,
    test_loader,
    device
)

predictions = torch.sigmoid(torch.Tensor(logits)).to("cpu").numpy()
labels = np.array(labels)

best_f1 = max(checkpoint["validation_f1"])
best_f1_idx = checkpoint["validation_f1"].index(best_f1)
threshold = checkpoint["best_thresholds"][best_f1_idx]

predicted_classes = (predictions >= threshold).astype(int)

# Calculate test metrics
test_accuracy = accuracy_score(labels, predicted_classes)
test_f1 = f1_score(labels, predicted_classes)
test_precision = precision_score(labels, predicted_classes)
test_recall = recall_score(labels, predicted_classes)

print("\nTest Results:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")


# Confusion matrix
cec_plot.plot_confusion_matrix(
    labels,
    predicted_classes,
    classes=["No Event", "Event"],
)

# Probability distribution
cec_plot.prob_distribution(
    labels,
    predictions,
)

# Additional analysis: prediction confidence
print("\nPrediction confidence analysis:")
confident_predictions = np.abs(predictions - 0.5) > 0.3
print(" Confident predictions: "
      f"{confident_predictions.sum()}/{len(predictions)} "
      f"({100 * confident_predictions.mean():.1f}%)")

if confident_predictions.sum() > 0:
    confident_accuracy = accuracy_score(
        labels[confident_predictions],
        predicted_classes[confident_predictions]
    )
    print(f"Accuracy on confident predictions: {confident_accuracy:.4f}")

# Class-wise performance
print("\nClass-wise performance:")
for class_label in [0, 1]:
    class_mask = labels == class_label
    if class_mask.sum() > 0:
        class_accuracy = accuracy_score(
            labels[class_mask],
            predicted_classes[class_mask]
        )
        class_mean_prob = predictions[class_mask].mean()
        print(f"  Class {class_label}: accuracy={class_accuracy:.4f}, "
              f"mean_prob={class_mean_prob:.4f}, n={class_mask.sum()}")

print("\nTesting completed!")
