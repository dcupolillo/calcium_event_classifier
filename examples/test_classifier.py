""" Created on Tue Jul  8 13:16:36 2025
    @author: dcupolillo """

import calcium_event_classifier as cec
import calcium_event_classifier.core.plot as cec_plot
from pathlib import Path
import flammkuchen as fl
import torch
import numpy as np

device = cec.set_device()

# Load model
model_path = Path(
    # r"models/unfiltered_model.pth")
    r"C:/Users/dcupolillo/Projects/calcium_event_classifier/models/251114_model_dff.pth")
checkpoint = torch.load(model_path)

classifier = cec.CalciumEventClassifier(
    dropout=checkpoint['hyperparams']["dropout"],
    out_channels_conv1=checkpoint['hyperparams']["conv1_channels"],
    out_channels_conv2=checkpoint['hyperparams']["conv2_channels"],
    out_channels_conv3=checkpoint['hyperparams']["conv3_channels"],
    out_channels_conv4=checkpoint['hyperparams']["out_channels_conv4"],
    kernel_size_conv1=checkpoint['hyperparams']["kernel_size_conv1"],
    kernel_size_conv2=checkpoint['hyperparams']["kernel_size_conv2"],
    kernel_size_conv3=checkpoint['hyperparams']["kernel_size_conv3"],
    kernel_size_conv4=checkpoint['hyperparams']["kernel_size_conv4"],
    fc1_out_features=checkpoint['hyperparams']["fc1_out_features"],
    fc2_out_features=checkpoint['hyperparams']["fc2_out_features"],
    pooling_type=checkpoint['hyperparams']["pooling_type"],
    leaky_relu_negative_slope=checkpoint['hyperparams'][
        "leaky_relu_negative_slope"],
    num_groups=checkpoint["hyperparams"]["num_groups"]
).to(device)

for name, param in classifier.named_parameters():
    if param.requires_grad:
        print(f"{name} - shape: {param.shape}")
        print(param.data)

for name, param in classifier.named_parameters():
    if param.requires_grad:
        print(f"{name} | mean: {param.data.mean():.4f}, "
              f"std: {param.data.std():.4f}")

# Initialize test dataset
test_data_path = Path(
    r"datasets/test_dataset_250530_ratio_3_1.h5")
test_data = fl.load(test_data_path)

test_dataset = cec.ZScoreDffDataset(
    test_data,
    augment=False,
)

test_loader = cec.load_test_dataset(
    test_dataset,
    batch_size=checkpoint['hyperparams']["batch_size"],
    summary=False)

classifier.load_state_dict(checkpoint['model_state_dict'])

# Inference on test dataset
labels, logits = cec.get_predictions_and_labels(
    classifier,
    test_loader,
    device)

predictions = torch.sigmoid(torch.Tensor(logits)).to("cpu").numpy()
labels = np.array(labels)

# Binarize predictions using best F1 threshold
best_f1 = max(checkpoint["validation_f1"])
best_f1_idx = checkpoint["validation_f1"].index(best_f1)
threshold = checkpoint["best_thresholds"][best_f1_idx]
predicted_classes = (np.array(predictions) >= threshold).astype(int)

# Plot results
cec_plot.plot_confusion_matrix(
    labels,
    predicted_classes,
    classes=["No Event", "Event"])

# Inspect class distribution
cec_plot.prob_distribution(
    labels, predictions)
