""" Created on Tue Jul  8 14:46:08 2025
    @author: dcupolillo """

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import time
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_recall_curve, average_precision_score)
import numpy as np


def compute_l1_regularization(model) -> float:

    return sum(p.abs().sum() for p in model.parameters())


def train(
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model: nn.Module,
        criterion: torch.nn.modules.loss,
        device: str,
        epochs: int,
        learning_rate: float,
        lr_drop_factor: float,
        lr_drop_patience: int,
        lambda1: float,
        lambda2: float,
        patience: int,
) -> tuple:
    """
    Train a model with early stopping and optional latent feature extraction.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader for training data.
    valid_loader : DataLoader
        DataLoader for validation data.
    model : nn.Module
        PyTorch model to train.
    criterion: torch.nn.modules.loss,
        Loss function.
    device : str
        Device identifier ("cpu" or "cuda").
    epochs : int
        Maximum number of training epochs.
    learning_rate : float
        Initial learning rate for the optimizer.
    lambda1 : float
        L1 regularization weight.
    lambda2 : float
        L2 regularization weight (weight decay).
    patience : int
        Number of epochs with no improvement before early stopping.

    Returns
    -------
    model : nn.Module
        Best-performing model on the validation set.
    train_loss : list of float
        Training loss per epoch.
    validation_loss : list of float
        Validation loss per epoch.
    train_f1 : list of float
        F1 score on the training set per epoch.
    validation_f1 : list of float
        F1 score on the validation set per epoch.
    validation_precision : list of float
        Precision on the validation set per epoch.
    validation_recall : list of float
        Recall on the validation set per epoch.
    best_thresholds : list of float
        Optimal classification threshold per epoch.
    validation_auc_pr : list of float
        AUC-PR score per epoch.
    valid_epoch_features : list of torch.Tensor or None
        Latent features from validation set per epoch (if available).
    valid_epoch_labels : list of torch.Tensor or None
        Corresponding labels for validation features (if available).
    train_epoch_features : list of torch.Tensor or None
        Latent features from training set per epoch (if available).
    train_epoch_labels : list of torch.Tensor or None
        Corresponding labels for training features (if available).
    """

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=lambda2,
        amsgrad=True
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_drop_factor,
        patience=lr_drop_patience,
        verbose=True
    )

    train_loss, validation_loss = [], []
    train_f1, validation_f1 = [], []
    validation_precision, validation_recall, best_thresholds = [], [], []
    validation_auc_pr = []

    valid_epoch_features = []
    valid_epoch_labels = []
    train_epoch_features = []
    train_epoch_labels = []

    best_loss = float('inf')
    best_model = copy.deepcopy(model)
    wait = 0

    print("======================================")
    print("         #### Training ####")
    print("======================================\n")
    start_time = time.time()

    for epoch in range(epochs):

        # Training
        model.train()
        running_loss = 0.0
        total_train_samples = 0
        all_preds = []
        all_targets = []
        all_train_feats = []

        # Looping through all the traces
        for data, target in tqdm(
                train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):

            if data.ndim == 2:
                data = data.unsqueeze(1).float()  # (B, T) → (B, 1, T)
            elif data.ndim == 3:
                data = data.float()  # already [B, C, T]

            # Change dtype before sending to device
            # data = data.float().unsqueeze(1).to(device)
            target = target.float().view(-1, 1)

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass
            features, logits = model(data, return_features=True)
            loss = criterion(logits, target)
            probs = torch.sigmoid(logits)

            l1_norm = compute_l1_regularization(model)
            loss += lambda1 * l1_norm
            # ...smoothing loss removed...

            running_loss += loss.item() * data.size(0)
            total_train_samples += data.size(0)

            loss.backward()  # Backpropagate
            optimizer.step()  # Update weights

            all_preds.extend(probs.detach().cpu().numpy().flatten())
            all_targets.extend(target.detach().cpu().numpy().flatten())
            all_train_feats.append(features.detach().cpu())

        # Compute average loss and prec./recall for training across batches
        avg_train_loss = running_loss / total_train_samples
        train_loss.append(avg_train_loss)

        precision, recall, thresholds = precision_recall_curve(
            all_targets, all_preds)

        precision_ = precision[1:]
        recall_ = recall[1:]

        f1_scores = 2 * (precision_ * recall_) / (precision_ + recall_ + 1e-8)

        if len(thresholds) > 0:
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
        else:
            # degenerate case: all_preds identical; fall back to 0.5
            best_idx = 0
            best_threshold = 0.5

        binarized_preds = (
            np.array(all_preds) >= best_threshold).astype(int)

        train_f1.append(
            f1_score(all_targets, binarized_preds, zero_division=0))

        train_epoch_features.append(torch.cat(all_train_feats))
        train_epoch_labels.append(torch.tensor(all_targets))

        # Validation
        model.eval()
        running_loss = 0.0
        total_valid_samples = 0
        all_preds = []
        all_targets = []
        all_valid_feats = []

        with torch.no_grad():  # Disable gradient calculation for validation

            for data, target in valid_loader:

                if data.ndim == 2:  # [B, T]
                    data = data.unsqueeze(1)  # → [B, 1, T]
                elif data.ndim == 3:
                    data = data  # already [B, C, T]

                target = target.float()
                target = target.view(-1, 1)
                data, target = data.to(device), target.to(device)

                features, logits = model(data, return_features=True)
                loss = criterion(logits, target)
                probs = torch.sigmoid(logits)

                running_loss += loss.item() * data.size(0)
                total_valid_samples += data.size(0)

                all_preds.extend(probs.detach().cpu().numpy().flatten())
                all_targets.extend(target.detach().cpu().numpy().flatten())
                all_valid_feats.append(features.cpu())

        # Compute average loss and prec./recall for validation
        avg_valid_loss = running_loss / total_valid_samples
        validation_loss.append(avg_valid_loss)

        precision, recall, thresholds = precision_recall_curve(
            all_targets, all_preds)

        precision_ = precision[1:]
        recall_ = recall[1:]

        f1_scores = 2 * (precision_ * recall_) / (precision_ + recall_ + 1e-8)

        if len(thresholds) > 0:
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            val_f1 = f1_scores[best_idx]
        else:
            best_idx = 0
            best_threshold = 0.5
            # compute F1 at this fixed threshold for logging
            bin_preds = (np.array(all_preds) >= best_threshold).astype(int)
            val_f1 = f1_score(all_targets, bin_preds, zero_division=0)

        val_auc_pr = average_precision_score(all_targets, all_preds)

        validation_f1.append(val_f1)
        validation_precision.append(precision[best_idx])
        validation_recall.append(recall[best_idx])
        best_thresholds.append(best_threshold)

        validation_auc_pr.append(val_auc_pr)

        valid_epoch_features.append(torch.cat(all_valid_feats))
        valid_epoch_labels.append(torch.tensor(all_targets))

        # Eventually, reduces learning rate
        scheduler.step(avg_valid_loss)

        # Early Stopping Check based on accuracy
        if avg_valid_loss < best_loss:  # check if new best loss
            best_loss = avg_valid_loss
            best_model = copy.deepcopy(model)
            wait = 0
        else:
            # if loss is not improving, increase the wait until patience
            wait += 1

        if wait > patience:
            print(f"Early stopping at epoch: {epoch+1}")
            break

        print(
            f"Epoch {epoch+1}: "
            f"🔵 Train loss: {avg_train_loss:.2f} | "
            f"🟢 Val loss: {avg_valid_loss:.2f} | "
            f"🎯 Val F1: {validation_f1[-1]:.3f} @ thr={best_threshold:.2f} | "
            f"🔻 Prec: {validation_precision[-1]:.3f}, "
            f"Rec: {validation_recall[-1]:.3f} | "
            f"🏆 Best loss: {best_loss:.2f}"

        )

    end_time = time.time()
    training_time = (
        time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))
    print(f"\n🕒 Time for training: {training_time}")
    print("======================================\n")

    return (
        best_model,
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
        train_epoch_labels
    )
