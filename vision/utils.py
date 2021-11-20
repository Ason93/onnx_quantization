#-*-conding:utf-8-*-
import numpy as np

def has_negative(tensor):
    '''
    Check the tensor has the negative value or not.

    :param tensor: input ndarray data.
    :return: Bool value. True means the tensor has the
             negative value.
    '''
    return True if np.amin(tensor) < 0 else False

def single_trans(tensor, axis=0):
    '''
    Transpose tensor one dim for each time.

    :param tensor: input ndarray data.
    :param axis: which dim put first.
    :return: new tensor.
    '''
    if axis > len(tensor.shape) - 1:
        print("Axis can't smaller than dims of tensor.")
        return None
    if axis == 0:
        return tensor
    temp_list = []
    for i in range(len(tensor.shape)):
        temp_list.append(i)
    temp_list.pop(axis)
    trans_list = [axis]
    trans_list.extend(temp_list)
    return tensor.transpose(trans_list)    

def tensor_maxes_std(tensor):
    '''
    Check the tensor per channel maxes std.
   
    :param tensor: input ndarray data.
    :return: std. of tensor per channel maxes.
    '''
    c_maxes = []
    for index in range(tensor.shape[0]):
        c_max = np.max(np.abs(tensor[index]))
        c_maxes.append(c_max)
    return np.std(c_maxes)

def tensor_maxes_ratio(tensor):
    '''
    Check the tensor per channel maxes std.

    :param tensor: input ndarray data.
    :return: max-min ratio of tensor per channel maxes.
    '''
    c_maxes = []
    for index in range(tensor.shape[0]):
        c_max = np.max(np.abs(tensor[index]))
        c_maxes.append(c_max)
    return np.max(c_maxes) * 1.0 / np.min(c_maxes)


def tensor_compute_mse(tensor, o_max, bit_num = 8):
    '''
    Compute mse based on tensor and optimized max.
   
    :param tesnor: input ndarray data.
    :param o_max: optimized max od tensor.
    :return: mse
    '''
    tensor = tensor.ravel()
    b = np.mean(np.abs(tensor - np.mean(tensor)))
    if has_negative(tensor):
        mse = 2 * (b ** 2) * ((np.e) ** (-o_max / b)) + ((o_max ** 2) / (3 * (2 ** (2 * bit_num))))
    else:
        mse = (b ** 2) * ((np.e) ** (-o_max / b)) + ((o_max ** 2) / (24 * (2 ** (2 * bit_num))))
    return mse