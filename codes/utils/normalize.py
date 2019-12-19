import numpy as np


def f(x, w, b):
    return w * x + b


def get_line(min_value, max_value):
    w = 1 / (max_value - min_value)
    b = -w * min_value
    return w, b


def linear_interpolation(patch):
    w, b = get_line(np.min(patch), np.max(patch))
    transformed_patch = f(patch, w, b)
    return transformed_patch


def standard_norm(patch):
    eps = 1e-06
    m = patch.mean()
    s = patch.std()
    normed_patch = (patch - m) / (s + eps)
    return normed_patch


def normalize(cur_frame, frame_size, num_channel, patch_size):
    num_step = frame_size // patch_size + 1
    for channel in range(num_channel):
        frame_slice = cur_frame[:, :, channel]
        for x_step in range(num_step):
            if x_step == num_step - 1:
                x_range = slice(x_step * patch_size, frame_size)
            else:
                x_range = slice(x_step * patch_size, (x_step + 1) * patch_size)
            for y_step in range(num_step):
                if y_step == num_step - 1:
                    y_range = slice(y_step * patch_size, frame_size)
                else:
                    y_range = slice(y_step * patch_size, (y_step + 1) * patch_size)
                patch = frame_slice[x_range, y_range]
                transformed_patch = standard_norm(patch)
                cur_frame[x_range, y_range, channel] = transformed_patch
                # patch_mean = np.mean(patch)
                # patch /= patch_mean
