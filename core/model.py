""" Created on Wed Oct 16 10:58:40 2024
    @author: dcupolillo """

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZScoreClassifier(nn.Module):

    def __init__(
            self,
            trace_length: int = 50,
            input_channels: int = 1,
            out_channels_conv1: int = 16,
            out_channels_conv2: int = 32,
            out_channels_conv3: int = 48,
            out_channels_conv4: int = 64,
            kernel_size_conv1: int = 9,
            kernel_size_conv2: int = 7,
            kernel_size_conv3: int = 5,
            kernel_size_conv4: int = 3,
            kernel_size_pool: int = 2,
            fc1_out_features: int = 128,
            fc2_out_features: int = 64,
            dropout: float = 0.5,
            pooling_type: str = "avg",
            leaky_relu_negative_slope: float = 0.01
    ) -> None:

        if pooling_type not in ["avg", "max"]:
            raise ValueError("Unrecognised pool type.")

        super().__init__()

        self.negative_slope = leaky_relu_negative_slope

        # Convolutional block 1
        self.conv1 = nn.Conv1d(
            input_channels,
            out_channels_conv1,
            kernel_size=kernel_size_conv1,
            padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels_conv1)

        # Convolutional block 2
        self.conv2 = nn.Conv1d(
            out_channels_conv1,
            out_channels_conv2,
            kernel_size=kernel_size_conv2,
            padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels_conv2)

        # Convolutional block 3
        self.conv3 = nn.Conv1d(
            out_channels_conv2,
            out_channels_conv3,
            kernel_size=kernel_size_conv3,
            padding="same")
        self.bn3 = nn.BatchNorm1d(out_channels_conv3)

        # Convolutional block 4
        self.conv4 = nn.Conv1d(
            out_channels_conv3,
            out_channels_conv4,
            kernel_size=kernel_size_conv4,
            padding="same")
        self.bn4 = nn.BatchNorm1d(out_channels_conv4)

        if pooling_type == "avg":
            self.pool = nn.AvgPool1d(kernel_size=kernel_size_pool)
        else:
            self.pool = nn.MaxPool1d(kernel_size=kernel_size_pool)

        # Fully Connected Layers
        self.fc1 = nn.Linear(
            out_channels_conv4 * ((trace_length // 2) // 2), fc1_out_features)
        self.fc2 = nn.Linear(fc1_out_features, fc2_out_features)
        self.fc3 = nn.Linear(fc2_out_features, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            return_features: bool = False
    ) -> torch.Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, self.negative_slope)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, self.negative_slope)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = self.dropout(x)

        features = x

        x = self.fc3(x)
        output = torch.sigmoid(x)

        if return_features:
            return features, output
        return output
