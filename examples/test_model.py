""" Created on Tue Jul  8 13:16:36 2025
    @author: dcupolillo """

import zscore_classifier as zsc
import zscore_classifier.core.plot as zsc_plot
from pathlib import Path
import flammkuchen as fl
import torch
import numpy as np

device = zsc.set_device()

# Load model
model_path = Path(r"models/unfiltered_model.pth")
checkpoint = torch.load(model_path)

classifier = zsc.ZScoreClassifier(
    dropout=checkpoint['hyperparams']["dropout"],
    out_channels_conv1=checkpoint['hyperparams']["out_channels_conv1"],
    out_channels_conv2=checkpoint['hyperparams']["out_channels_conv2"],
    out_channels_conv3=checkpoint['hyperparams']["out_channels_conv3"],
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
).to(device)

# Initialize test dataset
test_data_path = Path(
    r"datasets/test_dataset_250530_ratio_3_1.h5")
test_data = fl.load(test_data_path)

test_dataset = zsc.ZScoreDataset(
    test_data,
    event_range=None,
    augment=False,
)

test_loader = zsc.load_test_dataset(
    test_dataset,
    batch_size=checkpoint['hyperparams']["batch_size"],
    summary=False)

classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()

# Inference on test dataset
labels, predictions = zsc.get_predictions_and_labels(
    classifier,
    test_loader,
    device)

# Binarize predictions using best F1 threshold
best_f1 = max(checkpoint["validation_f1"])
best_f1_idx = checkpoint["validation_f1"].index(best_f1)
threshold = checkpoint["best_thresholds"][best_f1_idx]
predicted_classes = (np.array(predictions) >= threshold).astype(int)

# Plot results
zsc_plot.plot_confusion_matrix(
    labels,
    predicted_classes,
    classes=["No Event", "Event"])
