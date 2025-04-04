import unittest
from functools import partial

from conv import MyConvStub, MyFilterStub
from blur_kernel import get_blur_kernel

import torch
from torch import nn
import torch.nn.functional as F

kernel_size_stub_constructor = partial(
    MyConvStub,
    input_channels=128,
    output_channels=128,
    num_groups=1,
    stride=1,
    dilation=1,
    bias=False,
)
filter_size_stub_constructor = partial(
    MyConvStub, kernel_size=(3, 3), num_groups=1, stride=1, dilation=1, bias=False
)
biased_filter_size_stub_constructor = partial(
    MyConvStub, kernel_size=(3, 3), num_groups=1, stride=1, dilation=1, bias=True
)
stride_size_stub_constructor = partial(
    MyConvStub,
    input_channels=128,
    output_channels=128,
    kernel_size=(3, 3),
    num_groups=1,
    dilation=1,
    bias=False,
)
dilation_size_stub_constructor = partial(
    MyConvStub,
    input_channels=128,
    output_channels=128,
    kernel_size=(3, 3),
    num_groups=1,
    stride=1,
    bias=False,
)
groups_size_stub_constructor = partial(
    MyConvStub,
    input_channels=128,
    output_channels=256,
    kernel_size=(3, 3),
    stride=1,
    dilation=1,
    bias=False,
)


ATOL = 1e-3


class ConvTests(unittest.TestCase):
    """A suite of test that performs numerical checking on ConvStub"""

    def test_square_kernels(self):
        kernel_sizes = [(k, k) for k in range(1, 8)]
        for kernel_size in kernel_sizes:
            # create a conv stub
            conv = kernel_size_stub_constructor(kernel_size=kernel_size)
            random_input = torch.randn(4, 128, 64, 64)

            output = conv.forward(random_input)
            test_output = F.conv2d(
                input=random_input,
                weight=conv.weight,
                groups=conv.groups,
                stride=conv.stride,
                dilation=conv.dilation,
                padding=0,
            )

            self.assertTrue(torch.allclose(output, test_output, atol=ATOL))

    def test_horizontal_kernel(self):
        kernel_sizes = [(k, k + 2) for k in range(1, 8)]
        for kernel_size in kernel_sizes:
            # create a conv stub
            conv = kernel_size_stub_constructor(kernel_size=kernel_size)
            random_input = torch.randn(4, 128, 64, 64)

            output = conv.forward(random_input)
            test_output = F.conv2d(
                input=random_input,
                weight=conv.weight,
                groups=conv.groups,
                stride=conv.stride,
                dilation=conv.dilation,
                padding=0,
            )

            self.assertTrue(torch.allclose(output, test_output, atol=ATOL))

    def test_vertical_kernel(self):
        kernel_sizes = [(k + 2, k) for k in range(1, 8)]
        for kernel_size in kernel_sizes:
            # create a conv stub
            conv = kernel_size_stub_constructor(kernel_size=kernel_size)
            random_input = torch.randn(4, 128, 64, 64)

            output = conv.forward(random_input)
            test_output = F.conv2d(
                input=random_input,
                weight=conv.weight,
                groups=conv.groups,
                stride=conv.stride,
                dilation=conv.dilation,
                padding=0,
            )

            self.assertTrue(torch.allclose(output, test_output, atol=ATOL))

    def test_increasing_filter_sizes(self):
        input_sizes = [(2**i) for i in range(0, 8)]
        output_sizes = [(3**i) for i in range(1, 6)]
        for input_size, output_size in zip(input_sizes, output_sizes):
            # create a conv stub
            conv = filter_size_stub_constructor(
                input_channels=input_size, output_channels=output_size
            )
            random_input = torch.randn(4, input_size, 64, 64)

            output = conv.forward(random_input)
            test_output = F.conv2d(
                input=random_input,
                weight=conv.weight,
                groups=conv.groups,
                stride=conv.stride,
                dilation=conv.dilation,
                padding=0,
            )

            self.assertTrue(torch.allclose(output, test_output, atol=ATOL))

    def test_decreasing_filter_sizes(self):
        output_sizes = [(2**i) for i in range(0, 8)]
        input_sizes = [(3**i) for i in range(1, 6)]
        for input_size, output_size in zip(input_sizes, output_sizes):
            # create a conv stub
            conv = filter_size_stub_constructor(
                input_channels=input_size, output_channels=output_size
            )
            random_input = torch.randn(4, input_size, 64, 64)

            output = conv.forward(random_input)
            test_output = F.conv2d(
                input=random_input,
                weight=conv.weight,
                groups=conv.groups,
                stride=conv.stride,
                dilation=conv.dilation,
                padding=0,
            )

            self.assertTrue(torch.allclose(output, test_output, atol=ATOL))

    def test_same_number_of_filters(self):
        input_sizes = [(2**i) for i in range(0, 8)]
        output_sizes = [(2**i) for i in range(0, 8)]
        for input_size, output_size in zip(input_sizes, output_sizes):
            # create a conv stub
            conv = filter_size_stub_constructor(
                input_channels=input_size, output_channels=output_size
            )
            random_input = torch.randn(4, input_size, 64, 64)

            output = conv.forward(random_input)
            test_output = F.conv2d(
                input=random_input,
                weight=conv.weight,
                groups=conv.groups,
                stride=conv.stride,
                dilation=conv.dilation,
                padding=0,
            )

            self.assertTrue(torch.allclose(output, test_output, atol=ATOL))

    def test_stride(self):
        strides = list(range(1, 4))
        for stride in strides:
            # create a conv stub
            conv = stride_size_stub_constructor(stride=stride)
            random_input = torch.randn(4, 128, 64, 64)

            output = conv.forward(random_input)
            test_output = F.conv2d(
                input=random_input,
                weight=conv.weight,
                groups=conv.groups,
                stride=conv.stride,
                dilation=conv.dilation,
                padding=0,
            )

            self.assertTrue(torch.allclose(output, test_output, atol=ATOL))

    def test_dilation(self):
        dilations = list(range(1, 4))
        for dilation in dilations:
            # create a conv stub
            conv = dilation_size_stub_constructor(dilation=dilation)
            random_input = torch.randn(4, 128, 64, 64)

            output = conv.forward(random_input)
            test_output = F.conv2d(
                input=random_input,
                weight=conv.weight,
                groups=conv.groups,
                stride=conv.stride,
                dilation=conv.dilation,
                padding=0,
            )

            self.assertTrue(torch.allclose(output, test_output, atol=ATOL))

    def test_grouping(self):
        groups = [4, 8, 16, 32, 64]
        for group in groups:
            # create a conv stub
            conv = groups_size_stub_constructor(num_groups=group)
            random_input = torch.randn(4, 128, 64, 64)

            output = conv.forward(random_input)
            test_output = F.conv2d(
                input=random_input,
                weight=conv.weight,
                groups=conv.groups,
                stride=conv.stride,
                dilation=conv.dilation,
                padding=0,
            )

            self.assertTrue(torch.allclose(output, test_output, atol=ATOL))

    def test_biased_conv(self):
        input_sizes = [(2**i) for i in range(0, 8)]
        output_sizes = [(3**i) for i in range(1, 6)]
        for input_size, output_size in zip(input_sizes, output_sizes):
            # create a conv stub
            conv = biased_filter_size_stub_constructor(
                input_channels=input_size, output_channels=output_size
            )
            random_input = torch.randn(4, input_size, 64, 64)

            output = conv.forward(random_input)
            test_output = F.conv2d(
                input=random_input,
                weight=conv.weight,
                groups=conv.groups,
                stride=conv.stride,
                dilation=conv.dilation,
                bias=conv.bias,
                padding=0,
            )

            self.assertTrue(torch.allclose(output, test_output, atol=ATOL))

    def test_zero_biased_conv(self):
        input_sizes = [(2**i) for i in range(0, 8)]
        output_sizes = [(3**i) for i in range(1, 6)]
        for input_size, output_size in zip(input_sizes, output_sizes):
            # create a conv stub
            conv = filter_size_stub_constructor(
                input_channels=input_size, output_channels=output_size
            )
            random_input = torch.randn(4, input_size, 64, 64)

            output = conv.forward(random_input)
            test_output = F.conv2d(
                input=random_input,
                weight=conv.weight,
                groups=conv.groups,
                stride=conv.stride,
                dilation=conv.dilation,
                bias=torch.zeros(output_size),
                padding=0,
            )

            self.assertTrue(torch.allclose(output, test_output, atol=ATOL))


class FilterTest(unittest.TestCase):
    """A suite of tests that numerically checks the filtering operation"""

    class Filter2D(nn.Module):
        def __init__(self, channels: int, kernel: torch.Tensor):
            super().__init__()
            self.register_buffer(
                "filter", kernel[None, None, :, :].repeat(channels, 1, 1, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            return F.conv2d(
                x,
                self.filter,
                stride=1,
                padding=0,
                groups=c,
            )

    def test_filter(self):
        blur_sizes = list(range(1, 7))
        for blur_size in blur_sizes:
            # create a filter stub
            blur_kernel = get_blur_kernel(blur_size)
            conv = MyFilterStub(filter=blur_kernel, input_channels=128)
            test_conv = self.Filter2D(128, blur_kernel)
            random_input = torch.randn(4, 128, 64, 64)
            output = conv.forward(random_input)
            test_output = test_conv.forward(random_input)

            self.assertTrue(torch.allclose(output, test_output, atol=ATOL))


if __name__ == "__main__":
    print("\nRunning ConvTests:")
    conv_tests = ConvTests()

    print("\nTesting square kernels...")
    conv_tests.test_square_kernels()
    print("Square kernels test passed")

    print("\nTesting horizontal kernels...")
    conv_tests.test_horizontal_kernel()
    print("Horizontal kernels test passed")

    print("\nTesting vertical kernels...")
    conv_tests.test_vertical_kernel()
    print("Vertical kernels test passed")

    print("\nTesting increasing filter sizes...")
    conv_tests.test_increasing_filter_sizes()
    print("Increasing filter sizes test passed")

    print("\nTesting decreasing filter sizes...")
    conv_tests.test_decreasing_filter_sizes()
    print("Decreasing filter sizes test passed")

    print("\nTesting same number of filters...")
    conv_tests.test_same_number_of_filters()
    print("Same number of filters test passed")

    print("\nTesting stride...")
    conv_tests.test_stride()
    print("Stride test passed")

    print("\nTesting dilation...")
    conv_tests.test_dilation()
    print("Dilation test passed")

    print("\nTesting grouping...")
    conv_tests.test_grouping()
    print("Grouping test passed")

    print("\nTesting biased conv...")
    conv_tests.test_biased_conv()
    print("Biased conv test passed")

    print("\nTesting zero biased conv...")
    conv_tests.test_zero_biased_conv()
    print("Zero biased conv test passed")

    print("\nRunning FilterTest:")
    filter_test = FilterTest()

    print("\nTesting filter...")
    filter_test.test_filter()
    print("Filter test passed")
