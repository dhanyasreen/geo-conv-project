"""
conv_variants.py
-----------------
Compares two variants of the basic 2-D convolution operation on a
synthetic multi-scale geographic feature map:

    1. Standard Convolution        -> 3x3 kernel, dilation = 1
    2. Dilated (Atrous) Convolution -> 3x3 kernel, dilation = 3

Both use the SAME learnable-parameter budget (9 weights), which is the
whole point of the dilated variant: it enlarges the receptive field
WITHOUT adding parameters or downsampling the image.

Outputs written to ../results/:
    - standard_conv_output.png
    - dilated_conv_output.png
    - side_by_side_comparison.png
    - receptive_field_diagram.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_image():
    path = os.path.join(DATASET_DIR, "geo_image.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Run dataset/generate_dataset.py first to create geo_image.npy")
    return np.load(path)


def dilate_kernel(kernel, dilation):
    """Insert (dilation-1) zeros between kernel taps to build an
    effective larger kernel, exactly how a dilated convolution behaves."""
    k = kernel.shape[0]
    new_size = k + (k - 1) * (dilation - 1)
    dk = np.zeros((new_size, new_size), dtype=kernel.dtype)
    dk[::dilation, ::dilation] = kernel
    return dk


def conv2d(image, kernel, dilation=1, padding="same"):
    """Manual 2-D convolution (cross-correlation form, as used in CNNs)
    supporting a dilation factor, implemented with numpy stride tricks
    for full control/transparency (no black-box library call)."""
    eff_kernel = dilate_kernel(kernel, dilation)
    kh, kw = eff_kernel.shape

    if padding == "same":
        pad_h, pad_w = kh // 2, kw // 2
        img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    else:
        img = image

    H, W = img.shape
    out_h = H - kh + 1
    out_w = W - kw + 1

    shape = (out_h, out_w, kh, kw)
    strides = (img.strides[0], img.strides[1], img.strides[0], img.strides[1])
    windows = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

    out = np.einsum("ijkl,kl->ij", windows, eff_kernel)
    return out


def main():
    img = load_image()

    # A Laplacian-like edge/blob detecting 3x3 kernel (same weights used
    # for both variants so the comparison isolates the effect of dilation).
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1],
    ], dtype=np.float32)

    standard_out = conv2d(img, kernel, dilation=1)
    dilated_out = conv2d(img, kernel, dilation=3)

    eff_kernel_std = dilate_kernel(kernel, 1)
    eff_kernel_dil = dilate_kernel(kernel, 3)

    print("Standard convolution  -> kernel footprint: "
          f"{eff_kernel_std.shape}, trainable params: {kernel.size}")
    print("Dilated convolution   -> kernel footprint: "
          f"{eff_kernel_dil.shape}, trainable params: {kernel.size}")

    # ---------- Save individual outputs ----------
    plt.figure(figsize=(5, 5))
    plt.imshow(standard_out, cmap="inferno")
    plt.title("Standard Convolution (dilation=1)\nReceptive field 3x3")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "standard_conv_output.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.imshow(dilated_out, cmap="inferno")
    plt.title("Dilated Convolution (dilation=3)\nReceptive field 7x7")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "dilated_conv_output.png"), dpi=150)
    plt.close()

    # ---------- Side-by-side comparison (this doubles as som_map.png /
    # cluster_visualization.png equivalents for this project) ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input: Synthetic Geo Image\n(multi-scale features)")
    axes[0].axis("off")

    axes[1].imshow(standard_out, cmap="inferno")
    axes[1].set_title("Standard Conv Output\n(3x3 receptive field)")
    axes[1].axis("off")

    axes[2].imshow(dilated_out, cmap="inferno")
    axes[2].set_title("Dilated Conv Output\n(7x7 receptive field, same 9 params)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "side_by_side_comparison.png"), dpi=150)
    plt.close()

    # ---------- Receptive field diagram ----------
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(eff_kernel_std != 0, cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title(f"Standard 3x3\nfootprint {eff_kernel_std.shape}")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(eff_kernel_dil != 0, cmap="Greys", vmin=0, vmax=1)
    axes[1].set_title(f"Dilated 3x3 (rate=3)\nfootprint {eff_kernel_dil.shape}")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.suptitle("Effective Receptive Field: Standard vs Dilated Convolution")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "receptive_field_diagram.png"), dpi=150)
    plt.close()

    print("All result images saved to:", os.path.abspath(RESULTS_DIR))


if __name__ == "__main__":
    main()
