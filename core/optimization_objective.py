""" Created on Tue Jul  8 13:33:18 2025
    @author: dcupolillo """

import zscore_classifier as zsc
from optuna.trial import Trial
import torch
from pathlib import Path


def get_default_limits() -> dict:
    """
    Return default sampling ranges for all tunable hyperparameters.

    Returns
    -------
    dict
        Dictionary mapping each hyperparameter name to a tuple or list.
    """
    return {
        "learning_rate": (1e-5, 1e-2),
        "lambda1": (1e-5, 1e-2),
        "lambda2": (1e-6, 1e-3),
        "dropout": (0.0, 0.5),
        "augment_probability": (0.0, 0.5),
        "noise_level": (0.0, 0.5),
        "out_channels_conv1": (8, 32, 8),
        "out_channels_conv2": (16, 64, 16),
        "out_channels_conv3": (32, 96, 16),
        "out_channels_conv4": (48, 128, 16),
        "kernel_size_conv1": (3, 11, 2),
        "kernel_size_conv2": (3, 9, 2),
        "kernel_size_conv3": (3, 7, 2),
        "kernel_size_conv4": (3, 5, 2),
        "fc1_out_features": (24, 48, 8),
        "fc2_out_features": (24, 48, 8),
        "pooling_type": ["avg", "max"],
        "leaky_relu_negative_slope": (0.001, 0.1),
        "batch_size": (8, 32, 4),
        "random_seed": (0, 42, 7),
    }


def resolve(
    param: any,
    name: str,
    suggest_fn: callable,
    **kwargs
) -> any:
    """
    Helper function to resolve whether a value from Optuna is a tuple;
    otherwise return the fixed value.
    Allows to search space only for certain parameteres.

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


def define_search_space(
        trial: Trial,
        **params: dict
) -> dict:
    """
    Return a dictionary of hyperparameters, sampled or fixed.

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

    defaults = get_default_limits()
    for key in defaults:
        if key not in params:
            params[key] = defaults[key]

    return {
        "learning_rate":
            resolve(params["learning_rate"], "learning_rate",
                    trial.suggest_float, log=True),
        "lambda1":
            resolve(params["lambda1"], "lambda1",
                    trial.suggest_float, log=True),
        "lambda2":
            resolve(params["lambda2"], "lambda2",
                    trial.suggest_float, log=True),
        "dropout":
            resolve(params["dropout"], "dropout", trial.suggest_float),
        "augment_probability":
            resolve(params["augment_probability"], "augment_probability",
                    trial.suggest_float),
        "noise_level": resolve(params["noise_level"], "noise_level",
                               trial.suggest_float),
        "out_channels_conv1":
            resolve(params["out_channels_conv1"], "out_channels_conv1",
                    trial.suggest_int),
        "out_channels_conv2":
            resolve(params["out_channels_conv2"], "out_channels_conv2",
                    trial.suggest_int),
        "out_channels_conv3":
            resolve(params["out_channels_conv3"], "out_channels_conv3",
                    trial.suggest_int),
        "out_channels_conv4":
            resolve(params["out_channels_conv4"], "out_channels_conv4",
                    trial.suggest_int),
        "kernel_size_conv1":
            resolve(params["kernel_size_conv1"],
                    "kernel_size_conv1", trial.suggest_int),
        "kernel_size_conv2":
            resolve(params["kernel_size_conv2"], "kernel_size_conv2",
                    trial.suggest_int),
        "kernel_size_conv3":
            resolve(params["kernel_size_conv3"], "kernel_size_conv3",
                    trial.suggest_int),
        "kernel_size_conv4":
            resolve(params["kernel_size_conv4"], "kernel_size_conv4",
                    trial.suggest_int),
        "fc1_out_features":
            resolve(params["fc1_out_features"], "fc1_out_features",
                    trial.suggest_int),
        "fc2_out_features":
            resolve(params["fc2_out_features"], "fc2_out_features",
                    trial.suggest_int),
        "pooling_type":
            resolve(params["pooling_type"], "pooling_type",
                    trial.suggest_categorical),

        "leaky_relu_negative_slope":
            resolve(params["leaky_relu_negative_slope"],
                    "leaky_relu_negative_slope",
                    trial.suggest_float, log=True),
        "batch_size":
            resolve(params["batch_size"], "batch_size", trial.suggest_int),
        "random_seed":
            resolve(params["random_seed"], "random_seed", trial.suggest_int),
    }


def objective(
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
        reductions: list = [min]
) -> float:
    """
    Objective function for Bayesian optimization.

    Parameters
    ----------
    data : dict
        Dataset dictionary.
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

    Returns
    -------
    float
        Validation loss for the best epoch.
    """

    if search_space is None:
        search_space = get_default_limits()

    params = define_search_space(trial, search_space)
    train_fraction, validation_fraction = data_split

    dataset = zsc.ZScoreDataset(
        data,
        event_range=None,
        augment=True,
        augment_probability=params["augment_probability"],
        noise_level=params["noise_level"],
    )

    train_loader, valid_loader = zsc.split(
        dataset,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        batch_size=params["batch_size"],
        seed=params["random_seed"],
        summary=False
    )

    classifier = zsc.ZScoreClassifier(
        trace_length=len(dataset[0][0]),
        dropout=params["dropout"],
        out_channels_conv1=params["out_channels_conv1"],
        out_channels_conv2=params["out_channels_conv2"],
        out_channels_conv3=params["out_channels_conv3"],
        out_channels_conv4=params["out_channels_conv4"],
        kernel_size_conv1=params["kernel_size_conv1"],
        kernel_size_conv2=params["kernel_size_conv2"],
        kernel_size_conv3=params["kernel_size_conv3"],
        kernel_size_conv4=params["kernel_size_conv4"],
        fc1_out_features=params["fc1_out_features"],
        fc2_out_features=params["fc2_out_features"],
        pooling_type=params["pooling_type"],
        leaky_relu_negative_slope=params["leaky_relu_negative_slope"],
    ).to(device)

    trial.set_user_attr(
        "n_trainable_params", sum(p.numel() for p in classifier.parameters()))

    # Train model
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
    ) = zsc.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=classifier,
        criterion=criterion,
        device=device,
        epochs=epochs,
        learning_rate=params["learning_rate"],
        lambda1=params["lambda1"],
        lambda2=params["lambda2"],
        patience=patience
    )

    model_save_path = (
        Path(models_destination) /
        "optimization_models" /
        f"trial_{trial.number}_model.pth")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    trial.set_user_attr("model_path", model_save_path)

    results = {
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_f1": train_f1,
        "validation_f1": validation_f1,
        "validation_precision": validation_precision,
        "validation_recall": validation_recall,
        "best_thresholds": best_thresholds,
        "validation_auc_pr": validation_auc_pr,
        "hyperparams": params,
    }

    if trial_curves is not None:
        trial_curves[trial.number] = results

    torch.save({
        "model_state_dict": model.state_dict(),
        **results
    }, model_save_path)

    return [
        reduction(results[key])
        for key, reduction in zip(return_keys, reductions)]
