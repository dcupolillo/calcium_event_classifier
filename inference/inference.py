from pathlib import Path
import torch
import numpy as np
import calcium_event_classifier as cec


def load_classifier(model_path: str or Path) -> torch.nn.Module:
    """
    Load a pre-trained Z-Score neural network classifier.

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

    model = cec.CalciumEventClassifier2Ch2Ch(
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


def is_calcium_event(
        zscore: np.ndarray,
        dFF: np.ndarray,
        model: torch.nn.Module,
        device: str,
) -> tuple:
    """
    Perform inference on a single calcium trace using a trained model.

    Parameters
    ----------
    sweep : np.ndarray
        A 1D numpy array representing the calcium trace.
    model : torch.nn.Module
        A trained PyTorch model with an `input_channels` attribute.
    device : str
        Device to perform inference on ('cpu' or 'cuda').

    Returns
    -------
    float
        Predicted probability of a calcium event.
    """
    model.eval()

    zscore = torch.tensor(zscore, dtype=torch.float32)
    dFF = torch.tensor(dFF, dtype=torch.float32)

    dFF = (dFF - dFF.min()) / (dFF.max() - dFF.min() + 1e-8)

    trace = torch.stack([zscore, dFF], dim=0)
    trace = trace.unsqueeze(0).to(device)  # shape: (1, C, T)

    with torch.no_grad():
        logit = model(trace)
        prob = torch.sigmoid(logit)

    return logit.item(), prob.item()
