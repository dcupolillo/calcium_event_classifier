""" Created on Tue Jul  8 13:28:16 2025
    @author: dcupolillo """

import zscore_classifier as zsc
from zscore_classifier.core.optimization_objective import objective
import zscore_classifier.core.plot as zsc_plot
import optuna
from pathlib import Path
import flammkuchen as fl
import torch.nn as nn
import time

# Initialize device and random seed for reproducibility
device = zsc.set_device()
seed = zsc.set_seed(42)
train_fraction, validation_fraction = 0.75, 0.25
epochs = 300
patience = 8
bayesian_opt_trials = 30

# Loss function
criterion = nn.BCELoss()

# Initialize dataset
data_path = Path(
    r"neuralnetwork/zscore_decoder/datasets/bigger_zscore_labels.h5")
data = fl.load(data_path)

dataset = zsc.ZScoreDataset(data)

# Create optimization search
trial_curves = {}
study = optuna.create_study(directions=["minimize", "maximize"])

# Start search
start_time = time.time()

study.optimize(
    lambda trial: objective(
        data=data,
        data_split=(train_fraction, validation_fraction),
        device="cuda",
        criterion=criterion,
        epochs=epochs,
        patience=patience,
        trial=trial,
        trial_curves=trial_curves,
        return_keys=["validation_loss", "validation_f1"],
        reductions=[min, max]
    ),
    n_trials=bayesian_opt_trials)

training_time = (
    time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
print(f"🕒 Time for optimization: {training_time}")

# Plot search results
zsc_plot.plot_trial_metrics(trial_curves, [
    "train_loss",
    "validation_loss",
    "train_f1",
    "validation_f1",
    "validation_precision",
    "validation_recall"
])

zsc_plot.plot_pr_curves(trial_curves)
