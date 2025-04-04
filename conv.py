import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple


def get_conv_weight_and_bias(
    filter_size: Tuple[int, int],
    num_groups: int,
    input_channels: int,
    output_channels: int,
    bias: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert (
        input_channels % num_groups == 0
    ), "input channels must be divisible by groups number"
    assert (
        output_channels % num_groups == 0
    ), "output channels must be divisible by groups number"
    input_channels = input_channels // num_groups

    weight_matrix = torch.randn(output_channels, input_channels, *filter_size)
    if bias:
        bias_vector = torch.ones(output_channels)
    else:
        bias_vector = None
    return weight_matrix, bias_vector


class MyConvStub:
    def __init__(
        self,
        kernel_size: Tuple[int, int],
        num_groups: int,
        input_channels: int,
        output_channels: int,
        bias: bool,
        stride: int,
        dilation: int,
    ):
        self.weight, self.bias = get_conv_weight_and_bias(
            kernel_size, num_groups, input_channels, output_channels, bias
        )
        self.groups = num_groups
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, height, width = x.shape
        kernel_h, kernel_w = self.weight.shape[2:]

        # Calculate effective kernel size with dilation
        effective_kernel_h = (kernel_h - 1) * self.dilation + 1
        effective_kernel_w = (kernel_w - 1) * self.dilation + 1

        # Calculate output dimensions
        out_height = (height - effective_kernel_h) // self.stride + 1
        out_width = (width - effective_kernel_w) // self.stride + 1

        # Initialize output
        out_channels = self.weight.shape[0]
        output = torch.zeros(batch_size, out_channels, out_height, out_width)

        # Process each group
        in_channels_per_group = in_channels // self.groups
        out_channels_per_group = out_channels // self.groups

        for g in range(self.groups):
            # Get input and weight slices for this group
            input_slice = x[
                :, g * in_channels_per_group : (g + 1) * in_channels_per_group, :, :
            ]
            weight_slice = self.weight[
                g * out_channels_per_group : (g + 1) * out_channels_per_group, :, :, :
            ]

            # Process each batch
            for b in range(batch_size):
                # Process each output channel in this group
                for out_c in range(out_channels_per_group):
                    # Get the filters for this output channel
                    kernel = weight_slice[out_c]

                    # Compute output for each spatial position
                    for h in range(out_height):
                        for w in range(out_width):
                            h_start = h * self.stride
                            w_start = w * self.stride

                            # Extract input patch considering dilation
                            patch = input_slice[
                                b,
                                :,
                                h_start : h_start + effective_kernel_h : self.dilation,
                                w_start : w_start + effective_kernel_w : self.dilation,
                            ]

                            # Compute convolution for this position
                            output[b, g * out_channels_per_group + out_c, h, w] = (
                                patch * kernel
                            ).sum()

        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output


class MyFilterStub:
    def __init__(
        self,
        filter: torch.Tensor,
        input_channels: int,
    ):
        self.weight = filter
        self.input_channels = input_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        assert (
            channels == self.input_channels
        ), "Input channels must match specified channels"

        filter_h, filter_w = self.weight.shape
        out_height = height - filter_h + 1
        out_width = width - filter_w + 1

        # Create output tensor
        output = torch.zeros(batch_size, channels, out_height, out_width)

        # Apply filter to each channel independently
        for b in range(batch_size):
            for c in range(channels):
                # Compute each output position
                for h in range(out_height):
                    for w in range(out_width):
                        # Extract patch and apply filter
                        patch = x[b, c, h : h + filter_h, w : w + filter_w]
                        output[b, c, h, w] = (patch * self.weight).sum()

        return output
