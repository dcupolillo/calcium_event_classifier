from pathlib import Path
import torch
import numpy as np
import calcium_event_classifier as cec


def load_classifier(model_path: str or Path) -> torch.nn.Module:
    """
    Load a pre-trained Z-Score/dFF neural network classifier.

    Parameters
    ----------
    model_path : str
        Path to the saved PyTorch model file (.pth).

    Returns
    -------
    torch.nn.Module
        The loaded ZScoreNN model in evaluation mode.
    """
    checkpoint = torch.load(model_path)
    hyperparams = checkpoint['hyperparams']

    model = cec.CalciumEventClassifier2Ch(
        conv1_channels=hyperparams["conv1_channels"],
        conv2_channels=hyperparams["conv2_channels"],
        conv3_channels=hyperparams["conv3_channels"],
        conv1_kernel=hyperparams["conv1_kernel"],
        conv2_kernel=hyperparams["conv2_kernel"],
        conv3_kernel=hyperparams["conv3_kernel"],
        leaky_relu_negative_slope=hyperparams["leaky_relu_negative_slope"],
        dropout_rate=hyperparams["dropout_rate"],
        pool_kernel=hyperparams["pool_kernel"]
    )

    # Load weights and bias
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def load_classifier_dff(model_path: str or Path) -> torch.nn.Module:
    """
    Load a pre-trained DFF neural network classifier.

    Parameters
    ----------
    model_path : str
        Path to the saved PyTorch model file (.pth).

    Returns
    -------
    torch.nn.Module
        The loaded DffNN model in evaluation mode.
    """
    checkpoint = torch.load(model_path)
    hyperparams = checkpoint['hyperparams']
        

    model = cec.CalciumEventClassifierDff(
        conv1_channels=hyperparams["conv1_channels"],
        conv2_channels=hyperparams["conv2_channels"],
        conv3_channels=hyperparams["conv3_channels"],
        conv1_kernel=hyperparams["conv1_kernel"],
        conv2_kernel=hyperparams["conv2_kernel"],
        conv3_kernel=hyperparams["conv3_kernel"],
        leaky_relu_negative_slope=hyperparams["leaky_relu_negative_slope"],
        dropout_rate=hyperparams["dropout"],
        pool_kernel=hyperparams["pool_kernel"]
    )

    # Load weights and bias
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def is_calcium_event(
        zscore: np.ndarray,
        dFF: np.ndarray,
        model: torch.nn.Module,
        device: str,
) -> tuple:
    """
    Perform inference on a single calcium trace using a trained model.

    Automatically detects whether the model expects 2 channels (zscore + dFF)
    or 1 channel (dFF only) based on the model's class type.

    Parameters
    ----------
    zscore : np.ndarray
        A 1D numpy array representing the z-scored calcium trace.
    dFF : np.ndarray
        A 1D numpy array representing the ΔF/F₀ calcium trace.
    model : torch.nn.Module
        A trained PyTorch model (CalciumEventClassifier2Ch or
        CalciumEventClassifierDff).
    device : str
        Device to perform inference on ('cpu' or 'cuda').

    Returns
    -------
    tuple
        (logit, probability) - Raw model output and sigmoid probability.
    """
    model.eval()
    model.to(device)  # Ensure model is on the correct device

    # Detect model type by class name
    model_class_name = model.__class__.__name__
    is_two_channel = model_class_name == 'CalciumEventClassifier2Ch'

    if is_two_channel:
        zscore = torch.tensor(zscore, dtype=torch.float32)
        dFF = torch.tensor(dFF, dtype=torch.float32)
        dFF = (dFF - dFF.min()) / (dFF.max() - dFF.min() + 1e-8)
        trace = torch.stack([zscore, dFF], dim=0)
        trace = trace.unsqueeze(0).to(device)
    
    else:
        baseline = dFF[3:15]
        median = np.median(baseline)
        mad = np.median(np.abs(baseline - median)) + 1e-8
        dFF = (dFF - median) / mad

        dFF = torch.tensor(dFF, dtype=torch.float32)

        trace = dFF.unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logit = model(trace)
        prob = torch.sigmoid(logit)
    
    return logit.item(), prob.item()
