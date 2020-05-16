import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

def gaussian_filter(in_channel, out_channel, kernel_size=15, sigma=3):
    """
    This method returns 2d gaussian filter

    input :
    in_channel : Number of input channels
    out_channel : Expected number of output channels
    kernel_size : size of the filter (H x H)
    sigma : sigma

    output:
    returns : gaussian_filter
    """
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                  torch.exp(
                      -torch.sum((xy_grid - mean)**2., dim=-1) /\
                      (2*variance)
                  )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(in_channel, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                            kernel_size=kernel_size, groups=in_channel, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter