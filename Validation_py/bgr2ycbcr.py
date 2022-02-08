import torch
import os
import argparse
import time
import numpy as np
from compare import *

def convert_bgr_to_ycbcr(image):
    """Convert BGR image data to YCbCr format.
        using opencv 'nhwc', BGR format
    """
    begin = time.time()

    y = 16. + (64.738 * image[:, :, 2] + 129.057 * image[:, :, 1] + 25.064 * image[:, :, 0]) / 256.
    cb = 128. + (-37.945 * image[:, :, 2] - 74.494 * image[:, :, 1] + 112.439 * image[:, :, 0]) / 256.
    cr = 128. + (112.439 * image[:, :, 2] - 94.154 * image[:, :, 1] - 18.285 * image[:, :, 0]) / 256.
    output = np.array([y, cb, cr]).astype(np.float32)# nchw
    #output = output.transpose([1, 2, 0]) # nhwc
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

    input_c = np.fromfile(os.path.join(dir_path, 'Input_C_2'), dtype=np.uint8) # nhwc
    input_c = input_c.reshape(H, W, C)
    output_py = convert_bgr_to_ycbcr(input_c).flatten() # nchw

    output_c = np.fromfile(os.path.join(dir_path, 'Output_C_2'), dtype=np.float32) # nchw
    #output_c = output_c.reshape(C, H, W)
    #output_c = output_c.flatten()

    compare_two_tensor(output_py, output_c)
    #compare_two_tensor2(output_py.flatten(), output_c)
