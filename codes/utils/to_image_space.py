import numpy as np


def to_image_space(x):
    print(x.min())
    print(x.max())
    return np.uint8(255 * (x - x.min()) / (x.max() - x.min()))
