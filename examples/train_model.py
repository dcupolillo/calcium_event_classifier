""" Created on Tue Jul  8 12:49:18 2025
    @author: dcupolillo """

import zscore_classifier as zsc
from pathlib import Path
import flammkuchen as fl
import torch.nn as nn

# Initialize device and random seed for reproducibility
device = zsc.set_device()
seed = zsc.set_seed(42)

# Loss function
criterion = nn.BCELoss()

# Initialize dataset
data_path = Path(
    r"datasets/unfiltered_zscores.h5")
data = fl.load(data_path)

dataset = zsc.ZScoreDataset(data)

# Split dataset into training/validation sets
train_fraction, validation_fraction = 0.75, 0.25
batch_size = 12

train_loader, valid_loader = zsc.split(
    dataset,
    train_fraction=train_fraction,
    validation_fraction=validation_fraction,
    batch_size=batch_size,
    seed=seed,
    summary=False
)

classifier = zsc.ZScoreClassifier()

# Train model
epochs = 300
learning_rate = 1e-5
lambda1 = 0.001
lambda2 = 0.001
patience = 20

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
    epoch_features,
    epoch_labels,
) = zsc.train(
    train_loader=train_loader,
    valid_loader=valid_loader,
    model=classifier,
    criterion=criterion,
    device=device,
    epochs=epochs,
    learning_rate=learning_rate,
    lambda1=lambda1,
    lambda2=lambda2,
    patience=patience
)
