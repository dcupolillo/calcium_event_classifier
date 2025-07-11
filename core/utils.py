""" Created on Wed Oct 16 11:36:39 2024
    @author: dcupolillo """

import numpy as np
import torch
from torch.utils.data import (
    DataLoader, SubsetRandomSampler)
from sklearn.model_selection import train_test_split
import random


def set_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(
        seed=None,
        seed_torch=True
):

    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')

    return seed


def compute_class_weights(labels):
    """
    Compute class weights for handling class imbalance.
    Args:
    labels: List or tensor of all labels in the dataset.

    Returns:
    weights: Tensor of class weights for each sample.
    """
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    epsilon = 1e-9
    class_weights = total_samples / (class_counts + epsilon)
    pos_weight = class_weights[1] / class_weights[0]

    return torch.tensor([pos_weight])


def count_labels(labels, indices):
    label_counts = torch.bincount(labels[indices], minlength=2)
    return label_counts[0].item(), label_counts[1].item()


def split(
        dataset,
        train_fraction: float,
        validation_fraction: float,
        batch_size: int,
        summary: bool = True,
        seed: int = 42,
) -> tuple:

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    labels = torch.tensor(
        [int(dataset[i][1].item()) for i in indices], dtype=torch.long)

    assert round(train_fraction + validation_fraction) == 1.

    (
        train_indices,
        val_indices,
        train_labels,
        valid_labels
    ) = train_test_split(
         indices,
         labels,
         test_size=validation_fraction,
         stratify=labels,
         random_state=seed
     )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )
    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
    )

    if summary:
        train_0, train_1 = count_labels(labels, train_indices)
        val_0, val_1 = count_labels(labels, val_indices)

        total_samples = len(train_indices) + len(val_indices)
        print("----------------------------------")
        print("\n ==== Dataset Split Summary ====")
        print(f"Batch size: {batch_size}")
        print(f"Total samples: {dataset_size} | Assigned: {total_samples}")
        print(f"Train: {len(train_indices)} samples ({train_fraction:.2%}) "
              f" → {len(train_loader)} batches")
        print(f"   - Label 0: {train_0} ({train_0 / len(train_indices):.2%})"
              f" | Label 1: {train_1} ({train_1 / len(train_indices):.2%})")
        print(f"Validation: {len(val_indices)} samples ({validation_fraction:.2%})"
              f" → {len(valid_loader)} batches")
        print(f"   - Label 0: {val_0} ({val_0 / len(val_indices):.2%})"
              f" |  Label 1: {val_1} ({val_1 / len(val_indices):.2%})")

    return train_loader, valid_loader


def load_test_dataset(
        dataset,
        batch_size,
        summary):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    labels = torch.tensor(
        [int(dataset[i][1].item()) for i in indices], dtype=torch.long)

    sampler = SubsetRandomSampler(indices)

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler)

    if summary:
        test_0, test_1 = count_labels(labels, indices)

        print("----------------------------------")
        print("\n ==== Test Dataset Summary ====")
        print(f"Batch size: {batch_size}")
        print(f"Total samples: {dataset_size}")
        print(f"Test: {len(indices)} samples."
              f" → {len(test_loader)} batches")
        print(f"   - Label 0: {test_0} ({test_0 / len(indices):.2%})"
              f" | Label 1: {test_1} ({test_1 / len(indices):.2%})")

    return test_loader


def compute_l1_regularization(model):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_norm


def get_predictions_and_labels(
        model,
        data_loader,
        device,
):

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.float().unsqueeze(1).to(device)
            target = target.float().to(device)

            # Forward pass
            predicted = model.forward(data)

            y_true.extend(target.cpu().numpy().flatten())
            y_pred.extend(predicted.cpu().numpy().flatten())

    return y_true, y_pred


def evaluate_model_on_test(
        model,
        test_loader,
        device,
) -> tuple:

    model.eval()

    all_outputs = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.float().unsqueeze(1).to(device)

            output = model(data)
            all_outputs.append(output.cpu().numpy())

    all_outputs = np.concatenate(all_outputs).ravel()
    threshold = np.percentile(all_outputs, 99)

    (
     batched_data,
     batched_predictions,
     batched_labels,
     batched_raw_outputs) = [], [], [], []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.float().unsqueeze(1).to(device)
            labels = labels.float().to(device)

            # Forward pass
            output = model.forward(data)
            predicted = (output > threshold).float()

            # Store predictions and true labels grouped by batch
            batched_data.append(data.cpu().numpy())
            batched_predictions.append(predicted.cpu().numpy())
            batched_labels.append(labels.cpu().numpy())
            batched_raw_outputs.append(output.cpu().numpy())

    batched_predictions = [
        np.squeeze(arr) for arr in batched_predictions]
    batched_raw_outputs = [
        np.squeeze(arr) for arr in batched_raw_outputs]

    return (
        batched_data,
        batched_predictions,
        batched_labels,
        batched_raw_outputs)
