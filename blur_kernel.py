import torch
from torch import nn
import torch.nn.functional as F


def get_blur_kernel(size: int, ) -> torch.Tensor:
    def blur_values(size: int = 4) -> torch.Tensor:
        if size == 1:
            k = torch.tensor([1., ], dtype=torch.float32)
        elif size == 2:
            k = torch.tensor([1., 1.], dtype=torch.float32)
        elif size == 3:
            k = torch.tensor([1., 2., 1.], dtype=torch.float32)
        elif size == 4:
            k = torch.tensor([1., 3., 3., 1.], dtype=torch.float32)
        elif size == 5:
            k = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32)
        elif size == 6:
            k = torch.tensor([1., 5., 10., 10., 5., 1.], dtype=torch.float32)
        elif size == 7:
            k = torch.tensor([1., 6., 15., 20., 15., 6., 1.], dtype=torch.float32)
        return k

    def make_kernel_from_values(k: torch.Tensor) -> torch.Tensor:
        if k.ndim == 1:
            k = k[None, :] * k[:, None]

        k /= k.sum()

        return k

    k = blur_values(size)
    k = make_kernel_from_values(k)
    return k