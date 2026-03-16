'''
This package provides tools for analyzing and classifying calcium imaging data.
It includes modules for data preprocessing, model training, and inference.
'''
from .core.dffdataset import DffDataset
from .core.classifier_dff import CalciumEventClassifierDff
from .core.inference import is_calcium_event, load_classifier_dff
from .core.utils import (
    set_device, set_seed, split,
    load_test_dataset, evaluate_model_on_test,
    get_predictions_and_labels, extract_latent_features)
from .core.train_loop import train
from .build_dataset.GUI import run_app
from .build_dataset.manual_annotation_gui import run_manual_annotation

__version__ = "0.1.0"
__author__ = "Dario Cupolillo"

__all__ = [
    "DffDataset",
    "CalciumEventClassifierDff",
    "is_calcium_event",
    "load_classifier_dff",
    "set_device",
    "set_seed",
    "split",
    "load_test_dataset",
    "evaluate_model_on_test",
    "get_predictions_and_labels",
    "extract_latent_features",
    "train",
    "run_app",
    "run_manual_annotation",
    "__version__",
    "__author__",
]