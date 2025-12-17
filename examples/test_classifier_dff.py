""" Created on November 14, 2025
    @author: dcupolillo """

from pathlib import Path
import calcium_event_classifier as cec
import calcium_event_classifier.core.plot as cec_plot
from calcium_event_classifier.core.dffdataset import DffDataset
from calcium_event_classifier.core.classifier_dff import CalciumEventClassifierDff
import flammkuchen as fl
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, precision_recall_curve, confusion_matrix, auc)

device = cec.set_device()

# Load model
model_path = Path(
    r"C:/Users/dcupolillo/Projects/calcium_event_classifier/"
    r"models/251118_model_dff.pth")

checkpoint = torch.load(model_path, map_location=device)

# Print available keys in checkpoint
print(f"Checkpoint keys: {list(checkpoint.keys())}")

# Initialize classifier with saved hyperparameters
hyperparams = checkpoint['hyperparams']
print("\nModel hyperparameters:")
for key, value in hyperparams.items():
    print(f"  {key}: {value}")

# Initialize CalciumEventClassifierDff
classifier = CalciumEventClassifierDff(
    trace_length=50,  # Adjust if needed based on your data
    input_channels=hyperparams["input_channels"],
    conv1_channels=hyperparams["conv1_channels"],
    conv2_channels=hyperparams["conv2_channels"],
    conv3_channels=hyperparams["conv3_channels"],
    conv1_kernel=hyperparams["conv1_kernel"],
    conv2_kernel=hyperparams["conv2_kernel"],
    conv3_kernel=hyperparams["conv3_kernel"],
    leaky_relu_negative_slope=hyperparams["leaky_relu_negative_slope"],
    dropout_rate=hyperparams["dropout"],
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
    r"C:/Users/dcupolillo/Projects/calcium_event_classifier/"
    r"datasets/251114_test_dataset.h5")
test_data = fl.load(test_data_path)

test_dataset = DffDataset(
    test_data,
    augment=False,
)

test_loader = cec.load_test_dataset(
    test_dataset,
    batch_size=checkpoint['hyperparams']["batch_size"],
    summary=False,
    shuffle=False
)

labels, logits = cec.get_predictions_and_labels(
    classifier,
    test_loader,
    device
)

labels = np.array(labels)
predictions = torch.sigmoid(torch.Tensor(logits)).to("cpu").numpy()

# Calculate test metrics
precision, recall, thresholds = precision_recall_curve(labels, predictions)
precision_ = precision[1:]
recall_ = recall[1:]
f1_scores = 2 * (precision_ * recall_) / (precision_ + recall_ + 1e-8)

best_idx = f1_scores.argmax()
best_thr = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

predicted_classes = (predictions >= best_thr).astype(int)

test_accuracy = accuracy_score(labels, predicted_classes)
test_precision = precision_score(labels, predicted_classes)
test_recall = recall_score(labels, predicted_classes)
test_f1 = f1_score(labels, predicted_classes)

print("\nTest set performance:")
print(f" Optimal threshold: {best_thr:.4f}")
print(f" Accuracy: {test_accuracy:.4f}")
print(f" Precision: {test_precision:.4f}")
print(f" Recall: {test_recall:.4f}")
print(f" F1 Score: {test_f1:.4f}")

# Confusion matrix
cm = confusion_matrix(labels, predicted_classes)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)
print(f"  [[TN={tn:4d}  FP={fp:4d}]")
print(f"   [FN={fn:4d}  TP={tp:4d}]]")
print(f"\nConfusion Matrix Breakdown:")
print(f"  True Negatives (TN):  {tn:4d} - Correctly predicted no event")
print(f"  False Positives (FP): {fp:4d} - Incorrectly predicted event")
print(f"  False Negatives (FN): {fn:4d} - Missed actual events")
print(f"  True Positives (TP):  {tp:4d} - Correctly predicted event")
print(f"\nError rates:")
print(f"  False Positive Rate: {fp/(fp+tn)*100:.1f}% ({fp} out of {fp+tn} actual negatives)")
print(f"  False Negative Rate: {fn/(fn+tp)*100:.1f}% ({fn} out of {fn+tp} actual positives)")

# Class 0 metrics
class0_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
class0_recall = tn / (tn + fp) if (tn + fp) > 0 else 0

# Class 1 metrics  
class1_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
class1_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print("\nPer-class metrics:")
print(f"Class 0 (No Event):")
print(f"  Precision: {class0_precision:.3f} ({class0_precision*100:.1f}%)")
print(f"  Recall: {class0_recall:.3f} ({class0_recall*100:.1f}%)")
print(f"Class 1 (Event):")
print(f"  Precision: {class1_precision:.3f} ({class1_precision*100:.1f}%)")
print(f"  Recall: {class1_recall:.3f} ({class1_recall*100:.1f}%)")

# Plot confusion matrix and probability distribution
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

pr_auc = auc(recall, precision)
print(f"\nArea under PR curve: {pr_auc:.3f}")

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

# Sorted heatmap
sorted_idx = np.argsort(predictions)[::-1]
sorted_dff = test_data["dff"][sorted_idx]

plt.figure(figsize=(6, 10))
plt.imshow(sorted_dff, aspect="auto", cmap="viridis")
plt.colorbar(label="dFF")
plt.axvline(16, color="red", lw=1)
plt.xlabel("Time")
plt.ylabel("Traces (sorted by prob)")
plt.title("Sorted dFF heatmap (aligned)")
plt.tight_layout()
plt.show()

