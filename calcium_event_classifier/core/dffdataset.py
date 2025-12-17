""" Created on Thu Nov 14 2025
    @author: dcupolillo """

import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class DffDataset(Dataset):
    """
    Dataset class for dF/F calcium imaging data only.
    This class handles loading and augmenting the dFF data.
    """

    def __init__(
            self,
            data_dict: dict,
            augment: bool = True,
            augment_probability: float = 0.5,
            noise_level: float = 0.5,
            baseline_normalize: bool = True,
    ) -> None:
        """
        Initialize the dataset with data and augmentation options.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing 'dff' and 'label' keys.
        augment : bool
            Whether to apply data augmentation (adding noise).
        augment_probability : float
            Probability of applying augmentation to each sample.
        noise_level : float
            Standard deviation of the noise added during augmentation.
        baseline_normalize
            If True, apply per-trace baseline z-scoring using baseline_range.
        """

        self.dffs = torch.tensor(
            data_dict['dff'], dtype=torch.float32)
        self.labels = torch.tensor(
            data_dict["label"], dtype=torch.float32)

        self.augment = augment
        self.augment_probability = augment_probability
        self.noise_level = noise_level
        self.baseline_normalize = baseline_normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> tuple:

        dff = self.dffs[idx]
        label = self.labels[idx]

        if self.baseline_normalize:
            dff = self._baseline_normalize(dff)

        if (
                self.augment and
                torch.rand(1).item() < self.augment_probability
        ):
            dff = self.add_noise(dff)

        # Add channel dimension for single-channel input
        trace = dff.unsqueeze(0)

        return trace, label

    def _baseline_normalize(self, trace: torch.Tensor) -> torch.Tensor:
        """
        Z-score a trace using its baseline segment.
        """
        baseline = trace[3:15]
        median = np.median(baseline)
        mad = np.median(np.abs(baseline - median)) + 1e-8
        return (trace - median) / mad

    def get_labels(self) -> torch.Tensor:
        return self.labels

    def add_noise(self, data):
        noise = torch.zeros_like(data)
        baseline = data[3:15]

        baseline_med = torch.median(baseline)
        mad = torch.median(torch.abs(baseline - baseline_med)) + 1e-8

        noise_scale = self.noise_level * mad
        noise_baseline = torch.randn_like(baseline) * noise_scale
        noise[3:15] = noise_baseline

        return data + noise

    def count(self):
        return torch.bincount(self.labels.to(torch.long))

    def __str__(self):
        """
        Returns a string representation of the dataset,
        showing characteristics such as the
        number of samples, trace length, and label distribution.
        """
        num_samples = len(self)
        trace_length = self.dffs.shape[1]
        labels, counts = np.unique(self.labels, return_counts=True)
        percentages = counts / num_samples * 100

        characteristics = [
            "\n==== Dataset Characteristics ====",
            f"- Number of Samples: {num_samples}",
            f"- Trace Length: {trace_length}",
            "- Label Distribution:"
        ]
        for label, count, percent in zip(labels, counts, percentages):
            characteristics.append(
                f"  Label {label}: {count} instances | ({percent:.2f}%)")

        return "\n".join(characteristics)

    def plot(self):

        fig, axes = plt.subplots(1, 2, sharex=True)

        for dff, label in zip(self.dffs, self.labels):
            axes.flat[0 if label == 0 else 1].plot(
                dff, color="lightgray", alpha=0.5)

        axes.flat[0].plot(
            torch.mean(self.dffs[self.labels == 0], dim=0),
            color="mediumblue")
        axes.flat[1].plot(
            torch.mean(self.dffs[self.labels == 1], dim=0),
            color="crimson")

        for n in [0, 1]:
            axes.flat[n].set_ylim(0, 1)
            axes.flat[n].set_title(f"Label {n}")
            axes.flat[n].set_ylabel("dFF")
            axes.flat[n].set_xlabel("Data points")
