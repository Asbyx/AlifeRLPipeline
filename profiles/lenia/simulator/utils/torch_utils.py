import torch
import torch.nn.functional as F
from typing import Tuple


def unfold3d(
    tensor_padded: torch.Tensor, kernel_size: Tuple[int, int, int] = (3, 3, 3), stride: int = 1
) -> torch.Tensor:
    """
    Unfold a 4D tensor (B, C_pad, H_pad, W_pad) into patches of size kernel_size. The tensor should already be padded, according to the kernel size (use odd kernels, and pad kernel_size//2 on each side).

    Args:
        tensor_padded : torch.Tensor, input tensor of shape (B, C_pad, H_pad, W_pad)
        kernel_size : tuple of 3 integers (KC, KH, KW), size of the kernel to unfold
        stride : int, stride for unfolding
    Returns :
        Unfolded tensor of shape (B,C,KC,KH,KW,H,W)
    """
    if tensor_padded.ndim != 4:
        raise ValueError(
            f"Input tensor must be 4D (B, C_pad, H_pad, W_pad), but got shape {tensor_padded.shape}"
        )
    if len(kernel_size) != 3:
        raise ValueError(f"kernel_size must be a tuple of 3 integers (KC, KH, KW), but got {kernel_size}")

    kernel_C, kernel_H, kernel_W = kernel_size

    unfold_w = tensor_padded.unfold(dimension=3, size=kernel_W, step=stride)
    unfold_hw = unfold_w.unfold(dimension=2, size=kernel_H, step=stride)

    unfold_hw = unfold_hw.permute(0, 1, 2, 3, 5, 4)
    unfold_chw = unfold_hw.unfold(dimension=1, size=kernel_C, step=stride)
    unfold_chw = unfold_chw.permute(0, 1, 6, 4, 5, 2, 3)

    return unfold_chw


if __name__ == "__main__":
    # Test the function with a random tensor
    B, C_pad, H_pad, W_pad = 1, 3, 5, 5
    tensor = torch.ones(B, C_pad, H_pad, W_pad)

    print("Input Tensor Shape:", tensor.shape)  # Expected shape: (B, C_pad, H_pad, W_pad)
    kernel_size = (1, 2, 2)
    stride = 1

    unfolded_tensor = unfold3d(tensor, kernel_size=kernel_size, stride=stride)
    print(
        "Unfolded Tensor Shape:", unfolded_tensor.shape
    )  # Expected shape: (B, C_pad * kernel_C * kernel_H * kernel_W, num_patches)
