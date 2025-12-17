import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            leaky_relu_negative_slope: float,
    ) -> None:

        super().__init__()

        # Main convolutional layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same"
        )

        # Batch normalization
        self.bn = nn.BatchNorm1d(num_features=out_channels)

        # Activation
        self.act = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

        # Skip connection adjustment if channels don't match
        self.skip_connection = None
        if in_channels != out_channels:
            self.skip_connection = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,  # 1x1 conv for channel matching
                padding=0
            )
            self.skip_bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Store input for residual connection
        identity = x

        # Main path
        out = self.conv(x)
        out = self.bn(out)

        # Adjust skip connection if needed
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
            identity = self.skip_bn(identity)

        # Add residual connection
        out += identity
        out = self.act(out)

        return out


class CalciumEventClassifierDff(nn.Module):

    def __init__(
            self,
            trace_length: int = 50,
            input_channels: int = 1,
            conv1_channels: int = 32,
            conv2_channels: int = 32,
            conv3_channels: int = 64,
            conv1_kernel: int = 5,
            conv2_kernel: int = 3,
            conv3_kernel: int = 3,
            leaky_relu_negative_slope: float = 0.01,
            dropout_rate: float = 0.2,
            pool_kernel: int = 2
    ) -> None:

        super().__init__()

        self.trace_length = trace_length
        self.input_channels = input_channels
        self.negative_slope = leaky_relu_negative_slope

        # First residual block
        self.res_block1 = ResidualBlock(
            in_channels=input_channels,
            out_channels=conv1_channels,
            kernel_size=conv1_kernel,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
        )

        # Second residual block
        self.res_block2 = ResidualBlock(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            kernel_size=conv2_kernel,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
        )

        # Third residual block
        self.res_block3 = ResidualBlock(
            in_channels=conv2_channels,
            out_channels=conv3_channels,
            kernel_size=conv3_kernel,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
        )

        # Pooling and regularization
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Calculate flattened size after pooling
        pooled_length = trace_length // pool_kernel

        # Fully connected layer
        self.fc = nn.Linear(
            in_features=conv3_channels * pooled_length,
            out_features=1
        )

    def forward(
            self,
            x: torch.Tensor,
            return_features: bool = False
    ) -> torch.Tensor:

        # Three residual blocks before pooling
        x = self.res_block1(x)  # Now expects 1 channel (dFF only)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # Apply dropout and pooling after all conv layers
        x = self.dropout(x)
        x = self.pool(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        features = x
        output = self.fc(x)

        if return_features:
            return features, output
        return output
