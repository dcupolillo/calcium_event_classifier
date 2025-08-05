""" Created on Wed Oct 16 11:38:49 2024
    @author: dcupolillo """

import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class ZScoreDataset(Dataset):
    """
    Dataset class for Z-scored calcium imaging data.
    This class handles loading and augmenting the data.
    """

    def __init__(
            self,
            data_dict: dict,
            augment: bool = True,
            augment_probability: float = 0.5,
            noise_level: float = 0.5,
            normalize_dFF: bool = True,
    ) -> None:
        """
        Initialize the dataset with data and augmentation options.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing 'zscore', 'dff', and 'label' keys.
        augment : bool
            Whether to apply data augmentation (adding noise).
        augment_probability : float
            Probability of applying augmentation to each sample.
        noise_level : float
            Standard deviation of the noise added during augmentation.
        """

        self.zscores = torch.tensor(
            data_dict['zscore'], dtype=torch.float32)
        self.dffs = torch.tensor(
            data_dict['dff'], dtype=torch.float32)
        self.labels = torch.tensor(
            data_dict["label"], dtype=torch.float32)

        self.augment = augment
        self.augment_probability = augment_probability
        self.noise_level = noise_level

        if normalize_dFF:
            self.dffs = (
                self.dffs - self.dffs.min()
                ) / (self.dffs.max() - self.dffs.min() + 1e-8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> tuple:

        zscore = self.zscores[idx]
        dff = self.dffs[idx]
        label = self.labels[idx]

        if (self.augment and
                torch.rand(1).item() < self.augment_probability):
            zscore = self.add_noise(zscore)
            dff = self.add_noise(dff)

        trace = torch.stack([zscore, dff], dim=0)

        return trace, label

    def get_labels(self) -> torch.Tensor:
        return self.labels

    def add_noise(self, data):
        noise = torch.randn_like(data) * self.noise_level
        return data + noise

    def count(self):
        return torch.bincount(torch.tensor(self.labels))

    def __str__(self):
        """
        Returns a string representation of the dataset,
        showing characteristics such as the
        number of samples, trace length, and label distribution.
        """
        num_samples = len(self)
        trace_length = self.zscores.shape[1]
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

        fig, axes = plt.subplots(2, 2, sharex=True)

        for zscore, dff, label in zip(
                self.zscores, self.dffs, self.labels):


            axes.flat[0 if label == 0 else 1].plot(
                zscore, color="lightgray", alpha=0.5)
            axes.flat[2 if label == 0 else 3].plot(
                dff, color="lightgray", alpha=0.5)

        axes.flat[0].plot(
            torch.mean(self.zscores[self.labels == 0], dim=0),
            color="mediumblue")
        axes.flat[1].plot(
            torch.mean(self.zscores[self.labels == 1], dim=0),
            color="crimson")

        axes.flat[2].plot(
            torch.mean(self.dffs[self.labels == 0], dim=0),
            color="mediumblue")
        axes.flat[3].plot(
            torch.mean(self.dffs[self.labels == 1], dim=0),
            color="crimson")

        for n in [0, 1]:
            axes.flat[n].set_ylim(-2, 8)
            axes.flat[n].set_title(f"Label {n}")
        for n in [2, 3]:
            axes.flat[n].set_ylim(0, 1)

        axes.flat[0].set_ylabel("Z-score")
        axes.flat[2].set_ylabel("dFF0")
        axes.flat[2].set_xlabel("Data points")
