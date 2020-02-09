import numpy as np
from numpy import pi, exp, sqrt


def get_gaussian_filter(shape=(9, 9, 3), sigma=1):
    k = shape[0] // 2
    probs = np.array([exp(-z*z / (2*sigma*sigma)) / sqrt(2*pi*sigma*sigma) for z in range(-k, k + 1)])
    kernel_2d = np.outer(probs, probs)

    ch = shape[2] // 2
    prob_3d = probs[k - ch: k + ch + 1]
    kernel_3d = kernel_2d[:, :, None] * prob_3d

    return kernel_3d


def get_balance_filter(shape=(11,11, 3)):
    balance_filter = np.ones(shape) * -1
    balance_filter[1: 10, 1: 10, :] = 0.5
    return balance_filter


def get_2d_gaussian_filter(shape=(7, 7), sigma=1):
    k = shape[0] // 2
    probs = np.array([exp(-z * z / (2 * sigma * sigma)) / sqrt(2 * pi * sigma * sigma) for z in range(-k, k + 1)])
    kernel_2d = np.outer(probs, probs)
    return kernel_2d[..., None]
