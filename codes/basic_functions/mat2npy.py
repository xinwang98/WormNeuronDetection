import h5py
from scipy.io import loadmat
import numpy as np
import os


def mat2npy(index_and_position_path='./data/mat_files/index_and_position.mat',
            img_stack_path='./data/mat_files/img_Stack.mat',
            save_dir='./data/npy_files', num_frame=50, num_neuron=107):
    """
    This function change .mat files to .npy files and sort the neuron index for convenience
    """
    if os.path.exists(save_dir):
        return
    else:
        os.makedirs(save_dir)
        index_and_position = loadmat(index_and_position_path)
        neuron_index = index_and_position['neuron_index_data']
        neuron_position = index_and_position['neuron_position_data'][:, 0]

        ordered_neuron_position = np.zeros((num_frame, num_neuron, 3))
        for frame in range(num_frame):
            idx_list = neuron_index[frame, 0][0, :] - 1
            pos_list = neuron_position[frame]
            for neuron in range(num_neuron):
                real_idx,  = np.where(idx_list == neuron)
                ordered_neuron_position[frame, neuron, :] = pos_list[:, real_idx][:, 0] - 1

        np.save(os.path.join(save_dir, 'ordered_neuron_position.npy'), ordered_neuron_position)

        img_stack = h5py.File(img_stack_path)
        img_stack_keys = img_stack['img_Stack'][0, :]
        img_stack_extract = []
        for i in range(img_stack_keys.shape[0]):
            img_stack_extract.append(np.transpose(img_stack[img_stack_keys[i]]))     # matrix load by h5py would be transposed
        np.save(os.path.join(save_dir, 'img_stack.npy'), np.array(img_stack_extract))

        print('done!')
