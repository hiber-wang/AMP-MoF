import torch
import torch.fft as fft
import math

def fourier_filter(x, scale, d_s=0.25):
    dtype = x.dtype
    x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()

    for h in range(H):
        for w in range(W):
            d_square = (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2
            if d_square <= 2 * d_s:
                mask[..., h, w] = scale

    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    x_filtered = x_filtered.type(dtype)
    return x_filtered


def freq_mix(x, noise, LPF, dims=(-1,)):
    """
    Noise reinitialization for N-dimensional data.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
        dims: dimensions to perform FFT on (e.g., (-3, -2, -1) for 3D)
    """
    # FFT
    x_freq = fft.fftn(x, dim=dims)
    x_freq = fft.fftshift(x_freq, dim=dims)
    noise_freq = fft.fftn(noise, dim=dims)
    noise_freq = fft.fftshift(noise_freq, dim=dims)

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high  # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=dims)
    x_mixed = fft.ifftn(x_freq_mixed, dim=dims).real

    return x_mixed


def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25, dims=(-1,)):
    """
    Compute a Gaussian low pass filter mask for N-dimensional data.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
        dims: dimensions to apply the filter (e.g., 1D, 2D, 3D etc.)
    """
    size = [shape[d] for d in dims]
    mask = torch.zeros(shape)
    if d_s == 0 or d_t == 0:
        return mask

    ranges = [torch.linspace(-1, 1, s) for s in size]
    grid = torch.meshgrid(*ranges, indexing='ij')
    d_square = sum((g ** 2) for g in grid)

    # Gaussian filter equation
    mask[tuple(slice(None) for _ in dims)] = torch.exp(-d_square / (2 * d_s ** 2))
    return mask


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25, dims=(-1,)):
    """
    Compute a Butterworth low pass filter mask for N-dimensional data.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
        dims: dimensions to apply the filter (e.g., 1D, 2D, 3D etc.)
    """
    size = [shape[d] for d in dims]
    mask = torch.zeros(shape)
    if d_s == 0 or d_t == 0:
        return mask

    ranges = [torch.linspace(-1, 1, s) for s in size]
    grid = torch.meshgrid(*ranges, indexing='ij')
    d_square = sum((g ** 2) for g in grid)

    # Butterworth filter equation
    mask[tuple(slice(None) for _ in dims)] = 1 / (1 + (d_square / d_s ** 2) ** n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25, dims=(-1,)):
    """
    Compute an Ideal low pass filter mask for N-dimensional data.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
        dims: dimensions to apply the filter (e.g., 1D, 2D, 3D etc.)
    """
    size = [shape[d] for d in dims]
    mask = torch.zeros(shape)
    if d_s == 0 or d_t == 0:
        return mask

    ranges = [torch.linspace(-1, 1, s) for s in size]
    grid = torch.meshgrid(*ranges, indexing='ij')
    d_square = sum((g ** 2) for g in grid)

    # Ideal filter equation
    mask[tuple(slice(None) for _ in dims)] = (d_square <= d_s ** 2).float()
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25, dims=(-1,)):
    """
    Compute a Box low pass filter mask (approximated version) for N-dimensional data.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
        dims: dimensions to apply the filter (e.g., 1D, 2D, 3D etc.)
    """
    size = [shape[d] for d in dims]
    mask = torch.zeros(shape)
    if d_s == 0 or d_t == 0:
        return mask

    threshold_s = [round(s // 2 * d_s) for s in size]

    # Compute the center of each dimension
    center = [s // 2 for s in size]
    slices = [slice(c - t, c + t) for c, t in zip(center, threshold_s)]

    mask[tuple(slices)] = 1.0
    return mask
