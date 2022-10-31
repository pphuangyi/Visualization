"""
Turn grayscale image (like Toyzero) to colorful image
"""

#!/usr/bin/env python

import numpy as np
import torch


def symlog_tensor(tensor, linthresh=1.):
    """
    Calculate symlog of a tensor
    Input:
        - tensor (pytorch tensor): the input;
        - linthresh (float, default=1.): for number in [-linthresh, linthresh]
            will not be changed. For number outside the range, the symlog will
            be returned.
    Output:
        - symlog of the tensor with the given linthresh

    Note:
        The vesion of Symlog here is so calculated so that the
        piece-wise smooth function has a continuous first derivative.
    """

    assert linthresh > 0, 'linthresh must be greater than zero'

    tensor_abs = tensor.abs()
    tensor_symlog = torch.sign(tensor) * (torch.log(tensor_abs / linthresh) + 1) * linthresh
    tensor_symlog[tensor_abs <= linthresh] = tensor[tensor_abs <= linthresh]

    return tensor_symlog


def symlog_num(num, linthresh=1.):
    """
    Calculate symlog of a number
    Input:
        - num (float): the input;
        - linthresh (float, default=1.): for number in [-linthresh, linthresh]
            will not be changed. For number outside the range, the symlog will
            be returned.
    Output:
        - symlog of the number with the given linthresh
    Note:
        The vesion of Symlog here is so calculated so that the
        piece-wise smooth function has a continuous first derivative.
    """
    assert linthresh > 0, 'linthresh must be greater than zero'

    num_abs = abs(num)

    if num_abs <= linthresh:
        return num

    return torch.sign(num) * (torch.log(num_abs / linthresh) + 1) * linthresh


def normalize(tensor, vmin, vmax):
    """
    Input:
        - tensor: pytorch tensor;
        - vmin, vmax: the return will be (tensor - vmin) / (vmax - vmin);
    """
    return (tensor - vmin) / (vmax - vmin)


def gray_to_rgb(tensor, funcs):
    """
    Input:
        - tensor: pytorch tensor of shape (N, C, H, W) with C = 1;
        - funcs: dictionary with key {'r', 'g', 'b'}. The values
            are functions mapping [0, 1] to [0, 1]
    """
    assert tensor.shape[1] == 1, 'tensor x must be grayscale'
    vmin = tensor.min()
    vmax = tensor.max()
    assert (0 <= vmin <= 1.) and (0 <= vmax <= 1.), \
        'the input tensor must have values in [0, 1]'
    return torch.cat([funcs['r'](tensor), funcs['g'](tensor), funcs['b'](tensor)], dim=1)


def get_colormap(colormap):
    """
    Get color map functions mapping a grayscale tensor to rgb images
    Input:
        - colormap (str): string specifying the colormap to use
    """
    if colormap == 'bwr':
        func_r = lambda x: torch.clamp(2 * x, 0., 1.)
        func_g = lambda x: torch.min(2 * x, 2 * (1 - x))
        func_b = lambda x: torch.clamp(2 * (1 - x), 0., 1.)
    else:
        raise NotImplementedError(f'color map {colormap} is not implemented.')

    return {'r': func_r, 'g': func_g, 'b': func_b}


class Color:
    """
    Color a grayscale image to an rgb image
    """
    def __init__(self,
                 colormap='bwr',
                 symlog=True,
                 linthresh=1.,
                 vmin=None,
                 vmax=None,
                 vcenter=None):

        self.funcs = get_colormap(colormap)

        self.symlog = symlog
        self.linthresh = linthresh

        self.vmin = vmin
        self.vmax = vmax
        self.vcenter = vcenter

    def get_vrange(self, tensor):
        """
        Get vmin and vmax from initialization and/or tensor.
        """
        vmin = tensor.min() if self.vmin is None else self.vmin
        vmax = tensor.max() if self.vmax is None else self.vmax

        if self.vcenter is not None:
            vrange = max(abs(vmax - self.vcenter), abs(vmin - self.vcenter))
            vmin = self.vcenter - vrange
            vmax = self.vcenter + vrange

        return vmin, vmax

    def __call__(self, tensor):

        image = tensor.clone()

        vmin, vmax = self.get_vrange(image)
        image = torch.clamp(image, min=vmin, max=vmax)

        if self.symlog:
            image = symlog_tensor(image, linthresh=self.linthresh)
            vmin = symlog_num(vmin, self.linthresh)
            vmax = symlog_num(vmax, self.linthresh)

        image = normalize(image, vmin=vmin, vmax=vmax)

        return gray_to_rgb(image, self.funcs)
