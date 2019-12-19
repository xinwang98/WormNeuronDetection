import numpy as np


def nms(frame_after_conv, threshold, x_overlap, y_overlap, z_overlap, frame_size, num_channel):
    key_points_mask = (frame_after_conv >= threshold)
    key_points_mask[:, :512, :] = 0
    key_point_activation = np.where(key_points_mask == 1, frame_after_conv, 0)
    ite = 0
    while np.sum(key_point_activation) > 0:
        print('Iter {}:'.format(ite), np.sum(key_points_mask))
        ite += 1
        highest_activation_position = np.unravel_index(np.argmax(key_point_activation), key_point_activation.shape)
        hap = highest_activation_position
        if hap[0] - x_overlap >= 0:
            x_left = hap[0] - x_overlap
        else:
            x_left = 0
        if hap[0] + x_overlap <= frame_size:
            x_right = hap[0] + x_overlap
        else:
            x_right = frame_size
        if hap[1] - y_overlap >= 0:
            y_left = hap[1] - y_overlap
        else:
            y_left = 0
        if hap[1] + y_overlap <= frame_size:
            y_right = hap[1] + y_overlap
        else:
            y_right = frame_size
        if hap[2] - z_overlap >= 0:
            z_left = hap[2] - z_overlap
        else:
            z_left = 0
        if hap[2] + z_overlap <= num_channel:
            z_right = hap[2] + z_overlap
        else:
            z_right = num_channel

        key_point_activation[x_left: x_right, y_left: y_right, z_left: z_right] = 0
        key_points_mask[x_left: x_right, y_left: y_right, z_left: z_right] = 0
        key_points_mask[hap] = 1
    return key_points_mask
