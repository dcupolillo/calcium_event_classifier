""" Created on Wed Oct 16 11:38:49 2024
    @author: dcupolillo """

import torch
import numpy as np
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class ZScoreDataset(Dataset):

    def __init__(
            self,
            data_dict: dict,
            event_range: bool = False,
            augment: bool = True,
            augment_probability: float = 0.5,
            noise_level: float = 0.5,
            scaling: bool = False
    ) -> None:

        self.zscore_traces = data_dict['zscore']
        self.labels = data_dict['label']

        self.event_range = event_range
        self.augment = augment
        self.augment_probability = augment_probability
        self.noise_level = noise_level

        if scaling:
            scaler = MinMaxScaler(feature_range=(0, 1))
            self.zscore_traces = scaler.fit_transform(self.zscore_traces)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        trace = self.zscore_traces[idx]
        label = self.labels[idx]

        trace = torch.tensor(trace, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.augment:
            if torch.rand(1).item() < self.augment_probability:
                trace = self.add_noise(trace)
        if self.event_range:
            trace = self.extract_event(trace, self.event_range)

        return trace, label

    def add_noise(self, data):
        noise = torch.randn_like(data) * self.noise_level
        return data + noise

    def count(self):
        return torch.bincount(torch.tensor(self.labels))

    def extract_event(self, data, event_range):
        return data[event_range[0]:event_range[1]]

    def __str__(self):
        """
        Returns a string representation of the dataset,
        showing characteristics such as the
        number of samples, trace length, and label distribution.
        """
        num_samples = len(self)
        trace_length = (
            self.zscore_traces.shape[1]
            if len(self.zscore_traces.shape) > 1
            else "Variable")
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

        fig, ax = plt.subplots(1, 2)
        for trace, label in zip(self.zscore_traces, self.labels):
            ax[0 if label == 0 else 1].plot(trace)
