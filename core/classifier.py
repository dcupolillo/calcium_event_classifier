""" Created on Wed Oct 16 10:58:40 2024
    @author: dcupolillo """

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        negative_slope: float,
        num_groups: int
    ) -> None:

        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding="same")

        self.norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels
        )

        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

        # If input and output channels differ,
        # use a 1×1 convolution to project the input to match the output
        self.match_channels = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = self.match_channels(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)

        return out + identity


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
            leaky_relu_negative_slope: float = 0.01,
            num_groups: int = 4
    ) -> None:

        if pooling_type not in ["avg", "max"]:
            raise ValueError("Unrecognised pool type.")

        super().__init__()

        self.input_channels = input_channels
        self.negative_slope = leaky_relu_negative_slope

        self.block1 = ResidualBlock(
            input_channels, out_channels_conv1,
            kernel_size_conv1, negative_slope=leaky_relu_negative_slope,
            num_groups=num_groups
        )
        self.block2 = ResidualBlock(
            out_channels_conv1, out_channels_conv2,
            kernel_size_conv2, negative_slope=leaky_relu_negative_slope,
            num_groups=num_groups
        )
        self.block3 = ResidualBlock(
            out_channels_conv2, out_channels_conv3,
            kernel_size_conv3, negative_slope=leaky_relu_negative_slope,
            num_groups=num_groups
        )
        self.block4 = ResidualBlock(
            out_channels_conv3, out_channels_conv4,
            kernel_size_conv4, negative_slope=leaky_relu_negative_slope,
            num_groups=num_groups
        )

        self.pool = (
            nn.AvgPool1d(kernel_size_pool)
            if pooling_type == "avg"
            else nn.MaxPool1d(kernel_size_pool))

        reduced_length = trace_length
        for _ in range(2):  # Two pooling layers
            reduced_length = reduced_length // kernel_size_pool

        # Fully Connected Layers
        self.fc1 = nn.Linear(
            out_channels_conv4 * reduced_length, fc1_out_features)
        self.fc2 = nn.Linear(fc1_out_features, fc2_out_features)
        self.fc3 = nn.Linear(fc2_out_features, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            return_features: bool = False
    ) -> torch.Tensor:

        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = self.dropout(x)

        features = x

        x = self.fc3(x)
        output = x  # raw logits

        if return_features:
            return features, output
        return output
