""" Created on July 28, 2025
    @author: dcupolillo """

import zscore_classifier as zsc
from optuna.trial import Trial
import torch
import torch.nn as nn
from pathlib import Path
import optuna
from core.simpler_model import MinimalClassifier


def get_minimal_default_limits() -> dict:
    """
    Return optimized sampling ranges for MinimalClassifier hyperparameters.

    Returns
    -------
    dict
        Dictionary mapping each hyperparameter name to a tuple or list.
    """
    return {
        # Training parameters
        "learning_rate": (1e-5, 1e-2),
        "lr_drop_factor": (0.1, 0.7),
        "lr_drop_patience": (2, 8, 1),
        "lambda1": (1e-7, 1e-3),
        "lambda2": (1e-7, 1e-4),

        # Data augmentation
        "augment_probability": (0.0, 0.5),
        "noise_level": (0.01, 0.2),

        # Minimal architecture parameters
        "conv1_channels": (16, 64, 8),
        "conv2_channels": (16, 64, 8),
        "conv1_kernel": (3, 9, 2),
        "conv2_kernel": (3, 7, 2),
        "pool_kernel": (2, 4, 1),

        # Regularization
        "dropout_rate": (0.1, 0.5),
        "leaky_relu_negative_slope": (0.01, 0.1),

        # Training parameters
        "batch_size": (16, 128, 16),
        "random_seed": (0, 42, 7),
    }


def resolve_minimal(
        param: any,
        name: str,
        suggest_fn: callable,
        **kwargs
) -> any:
    """
    Helper function to resolve whether a value from Optuna is a tuple;
    otherwise return the fixed value.
    Allows to search space only for certain parameters.

    Parameters
    ----------
    param : tuple or any
        Tuple to define sampling range or fixed value.
    name : str
        Name of the hyperparameter.
    suggest_fn : callable
        Optuna suggest function (e.g. suggest_float, suggest_int).
    kwargs : dict
        Additional arguments passed to suggest_fn.

    Returns
    -------
    any
        Sampled or fixed hyperparameter value.
    """
    if isinstance(param, tuple):
        return suggest_fn(name, *param, **kwargs)
    else:
        return param


def define_minimal_search_space(
        trial: Trial,
        **params: dict
) -> dict:
    """
    Return a dictionary of hyperparameters for MinimalClassifier.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial used for sampling.
    params : dict
        Keyword arguments where each value is either a tuple (to sample)
        or fixed.

    Returns
    -------
    dict
        Dictionary of hyperparameter values.
    """
    defaults = get_minimal_default_limits()
    for key in defaults:
        if key not in params:
            params[key] = defaults[key]

    return {
        # Training parameters
        "learning_rate":
            resolve_minimal(params["learning_rate"], "learning_rate",
                            trial.suggest_float, log=True),
        "lr_drop_factor":
            resolve_minimal(params["lr_drop_factor"], "lr_drop_factor",
                            trial.suggest_float),
        "lr_drop_patience":
            resolve_minimal(params["lr_drop_patience"], "lr_drop_patience",
                            trial.suggest_int),
        "lambda1":
            resolve_minimal(params["lambda1"], "lambda1",
                            trial.suggest_float, log=True),
        "lambda2":
            resolve_minimal(params["lambda2"], "lambda2",
                            trial.suggest_float, log=True),

        # Data augmentation
        "augment_probability":
            resolve_minimal(params["augment_probability"],
                            "augment_probability",
                            trial.suggest_float),
        "noise_level":
            resolve_minimal(params["noise_level"], "noise_level",
                            trial.suggest_float),

        # Architecture parameters
        "conv1_channels":
            resolve_minimal(params["conv1_channels"], "conv1_channels",
                            trial.suggest_int),
        "conv2_channels":
            resolve_minimal(params["conv2_channels"], "conv2_channels",
                            trial.suggest_int),
        "conv1_kernel":
            resolve_minimal(params["conv1_kernel"], "conv1_kernel",
                            trial.suggest_int),
        "conv2_kernel":
            resolve_minimal(params["conv2_kernel"], "conv2_kernel",
                            trial.suggest_int),
        "pool_kernel":
            resolve_minimal(params["pool_kernel"], "pool_kernel",
                            trial.suggest_int),

        # Regularization
        "dropout_rate":
            resolve_minimal(params["dropout_rate"], "dropout_rate",
                            trial.suggest_float),
        "leaky_relu_negative_slope":
            resolve_minimal(params["leaky_relu_negative_slope"],
                            "leaky_relu_negative_slope",
                            trial.suggest_float, log=True),

        # Training parameters
        "batch_size":
            resolve_minimal(params["batch_size"], "batch_size",
                            trial.suggest_int),
        "random_seed":
            resolve_minimal(params["random_seed"], "random_seed",
                            trial.suggest_int),
    }


def minimal_objective(
        data: dict,
        data_split: tuple,
        device: str,
        criterion: torch.nn.modules.loss,
        epochs: int,
        patience: int,
        trial: Trial,
        models_destination: str or Path,
        search_space: dict = None,
        trial_curves: dict = None,
        return_keys: list = ["validation_loss"],
        reductions: list = [min],
        trace_length: int = 50
) -> float:
    """
    Objective function for MinimalClassifier Bayesian optimization.

    Parameters
    ----------
    data : dict
        Dataset dictionary with 'label' and 'zscore' keys.
    data_split : tuple
        (train_fraction, validation_fraction).
    device : str
        Torch device ("cpu" or "cuda").
    criterion : torch.nn.modules.loss._Loss
        Loss function.
    epochs : int
        Maximum number of training epochs.
    patience : int
        Early stopping patience.
    trial : optuna.trial.Trial
        Current Optuna trial.
    models_destination: str or Path
        The folder wherein a model for each trial will be saved.
    search_space : dict
        Custom search space bounds.
    trial_curves : dict
        Optional dict to store per-trial metrics.
    return_keys : list of str
        Metric names to return.
    reductions : list of callable
        Reduction function per return key (e.g. min, max).
    trace_length : int
        Length of input traces.

    Returns
    -------
    float
        Validation loss for the best epoch.
    """

    print("======================================")
    print(f"    #### Minimal Trial {trial.number} ####")
    print("======================================\n")

    if search_space is None:
        search_space = get_minimal_default_limits()

    params = define_minimal_search_space(trial, **search_space)
    train_fraction, validation_fraction = data_split

    # Create dataset with augmentation
    dataset = zsc.ZScoreDataset(
        data,
        event_range=None,
        augment=True,
        augment_probability=params["augment_probability"],
        noise_level=params["noise_level"],
    )

    # Split data
    train_loader, valid_loader = zsc.split(
        dataset,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        batch_size=params["batch_size"],
        seed=params["random_seed"],
        summary=False
    )

    # Calculate class weights for balanced training
    train_labels = torch.tensor(
        [int(label.item()) for _, label in train_loader.dataset],
        dtype=torch.float)

    num_pos = (train_labels == 1).sum().float()
    num_neg = (train_labels == 0).sum().float()
    pos_weight = num_neg / num_pos

    # Use weighted loss
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Initialize MinimalClassifier
    classifier = MinimalClassifier(
        trace_length=trace_length,
        conv1_channels=params["conv1_channels"],
        conv2_channels=params["conv2_channels"],
        conv1_kernel=params["conv1_kernel"],
        conv2_kernel=params["conv2_kernel"],
        leaky_relu_negative_slope=params["leaky_relu_negative_slope"],
        dropout_rate=params["dropout_rate"],
        pool_kernel=params["pool_kernel"]
    ).to(device)

    # Log number of parameters
    n_params = sum(p.numel() for p in classifier.parameters())
    trial.set_user_attr("n_trainable_params", n_params)
    print(f"Model has {n_params:,} trainable parameters")

    # Train model using existing training function
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
        epoch_labels
    ) = zsc.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=classifier,
        criterion=criterion,
        device=device,
        epochs=epochs,
        learning_rate=params["learning_rate"],
        lr_drop_factor=params["lr_drop_factor"],
        lr_drop_patience=params["lr_drop_patience"],
        lambda1=params["lambda1"],
        lambda2=params["lambda2"],
        patience=patience
    )

    # Evaluate model on validation set
    model.eval()
    n_samples = len(valid_loader.dataset)
    logits = torch.empty(n_samples, 1)

    with torch.no_grad():
        start = 0
        for x, _ in valid_loader:
            x = x.float().unsqueeze(1).to(device)
            out = model(x).cpu()
            end = start + out.size(0)
            logits[start:end] = out
            start = end

    probs = torch.sigmoid(logits)

    print(f"Logit range: {logits.min().item():.3f} to "
          f"{logits.max().item():.3f} | Median: {logits.median().item():.3f}")
    print(f"Prob  range: {probs.min().item():.3f} to "
          f"{probs.max().item():.3f} | Median: {probs.median().item():.3f}")
    print(f"Best validation AUC-PR: {validation_auc_pr[-1]:.4f}")
    print(f"Best validation F1: {max(validation_f1):.4f}")
    print("\n")

    # Save model
    model_save_path = (
        Path(models_destination) /
        "minimal_optimization_models" /
        f"trial_{trial.number}_minimal_model.pth")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    trial.set_user_attr("model_path", str(model_save_path))

    description = f"Minimal Trial {trial.number}: {params}"

    # Compile results
    results = {
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_f1": train_f1,
        "validation_f1": validation_f1,
        "validation_precision": validation_precision,
        "validation_recall": validation_recall,
        "best_thresholds": best_thresholds,
        "validation_auc_pr": validation_auc_pr,
        "epoch_features": epoch_features,
        "epoch_labels": epoch_labels,
        "hyperparams": params,
        "valid_probs": probs,
        "description": description,
        "n_params": n_params,
    }

    # Store trial curves if requested
    if trial_curves is not None:
        trial_curves[trial.number] = results

    # Save complete trial results
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_architecture": "MinimalClassifier",
        **results
    }, model_save_path)

    # Return requested metrics
    return [
        reduction(results[key])
        for key, reduction in zip(return_keys, reductions)]