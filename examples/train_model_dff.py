""" Created on Thu Nov 14 2025
    @author: dcupolillo """

import calcium_event_classifier as cec
from calcium_event_classifier.core.dffdataset import DffDataset
from calcium_event_classifier.core.classifier_dff import (
    CalciumEventClassifierDff)
from pathlib import Path
import flammkuchen as fl
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt


# Get today's date YYMMDD
today = datetime.now().strftime("%y%m%d")

# Initialize device and random seed for reproducibility
device = cec.set_device()
seed = cec.set_seed(1000)

# Loss function
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

# Initialize dataset
data_path = Path(
    r"C:/Users/dcupolillo/Projects/calcium_event_classifier/"
    r"datasets/251114_dataset.h5")
data = fl.load(data_path)

dataset = DffDataset(
    data,
    augment=False,
)

# Split dataset into training/validation sets
train_fraction, validation_fraction = 0.75, 0.25
batch_size = 16

train_loader, valid_loader = cec.split(
    dataset,
    train_fraction=train_fraction,
    validation_fraction=validation_fraction,
    batch_size=batch_size,
    seed=seed,
    summary=False)



input_channels = 1
conv1_channels = 16
conv2_channels = 32
conv3_channels = 64
conv1_kernel = 5
conv2_kernel = 3
conv3_kernel = 2
leaky_relu_negative_slope = 0.05
dropout_rate = 0.1
pool_kernel = 2

classifier = CalciumEventClassifierDff(
    trace_length=len(dataset[0][0][0]),  # Get trace length from first sample
    input_channels=input_channels,
    conv1_channels=conv1_channels,
    conv2_channels=conv2_channels,
    conv3_channels=conv3_channels,
    conv1_kernel=conv1_kernel,
    conv2_kernel=conv2_kernel,
    conv3_kernel=conv3_kernel,
    leaky_relu_negative_slope=leaky_relu_negative_slope,
    dropout_rate=dropout_rate,
    pool_kernel=pool_kernel,
    ).to(device)

# Train model
epochs = 500
learning_rate = 1e-5
lr_drop_factor = 0.5
lr_drop_patience = 5
lambda1 = 1e-5
lambda2 = 1e-4
patience = 8

(
    model,
    train_loss,
    validation_loss,
    train_f1,
    validation_f1,
    validation_precision,
    validation_recall,
    best_thresholds,
    validation_auc_pr,
    valid_epoch_features,
    valid_epoch_labels,
    train_epoch_features,
    train_epoch_labels,
) = cec.train(
    train_loader=train_loader,
    valid_loader=valid_loader,
    model=classifier,
    criterion=criterion,
    device=device,
    epochs=epochs,
    learning_rate=learning_rate,
    lr_drop_factor=lr_drop_factor,
    lr_drop_patience=lr_drop_patience,
    lambda1=lambda1,
    lambda2=lambda2,
    patience=patience,
)


save_path = Path(rf"{today}_model_dff.pth")

# Create dictionary to store
checkpoint = {
    "model_state_dict": model.state_dict(),
    "training_dataset": data_path,
    "train_loss": train_loss,
    "validation_loss": validation_loss,
    "train_f1": train_f1,
    "validation_f1": validation_f1,
    "validation_precision": validation_precision,
    "validation_recall": validation_recall,
    "best_thresholds": best_thresholds,
    "validation_auc_pr": validation_auc_pr,
    "valid_epoch_features": valid_epoch_features,
    "valid_epoch_labels": valid_epoch_labels,
    "train_epoch_features": train_epoch_features,
    "train_epoch_labels": train_epoch_labels,
    "hyperparams": {
        "learning_rate": learning_rate,
        "lr_drop_factor": lr_drop_factor,
        "lr_drop_patience": lr_drop_patience,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "epochs": epochs,
        "patience": patience,
        "input_channels": input_channels,
        "conv1_channels": conv1_channels,
        "conv2_channels": conv2_channels,
        "conv3_channels": conv3_channels,
        "conv1_kernel": conv1_kernel,
        "conv2_kernel": conv2_kernel,
        "conv3_kernel": conv3_kernel,
        "pool_kernel": pool_kernel,
        "dropout": dropout_rate,
        "leaky_relu_negative_slope": leaky_relu_negative_slope,
        "batch_size": batch_size,
        "random_seed": seed,
        "use_dff_only": True,
    }
}

# Save to disk
torch.save(checkpoint, save_path)

fig, ax = plt.subplots()
hyperparams = checkpoint["hyperparams"]
ax.plot(checkpoint["train_loss"])
ax.plot(checkpoint["validation_loss"])
ax.plot(checkpoint["train_f1"])
ax.plot(checkpoint["validation_f1"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss / F1 Score")
ax.set_title("Training and Validation Loss and F1 Score (dFF only)")
ax.legend(["Train Loss", "Validation Loss", "Train F1", "Validation F1"])
