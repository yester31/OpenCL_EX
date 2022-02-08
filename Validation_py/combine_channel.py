import torch
import os
import argparse
import time
import numpy as np
from compare import *

def combine_first_channel(input, input2):

    begin = time.time()

    output = np.concatenate((input2[:, :, :], input[1:, :, :]), axis=0)

    dur = time.time() - begin

    print('dur time(python) : %.3f [msec]' % (dur * 1000))
    return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='add parameters')

    parser.add_argument('--N', type=int, default=1, help='batch size')
    parser.add_argument('--C', type=int, default=3, help='channel')
    parser.add_argument('--H', type=int, default=1080, help='height')
    parser.add_argument('--W', type=int, default=1920, help='width')
    args = parser.parse_args()

    dir_path = os.path.dirname(__file__)

    N = args.N
    C = args.C
    H = args.H
    W = args.W

    input_c = np.fromfile(os.path.join(dir_path, 'Output_C_2'), dtype=np.float32) # nchw
    input_c = input_c.reshape(C, H, W)
    input_c2 = np.fromfile(os.path.join(dir_path, 'Output_C_4'), dtype=np.float32) # n1hw
    input_c2 = input_c2.reshape(1, H, W)

    output_py = combine_first_channel(input_c, input_c2) # nhwc

    output_c = np.fromfile(os.path.join(dir_path, 'Output_C_5'), dtype=np.float32) # nhwc

    #compare_two_tensor(output_py.flatten(), output_c)
    compare_two_tensor2(output_py.flatten(), output_c)
