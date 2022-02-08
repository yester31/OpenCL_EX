import torch
import os
import argparse
import time
import numpy as np
from compare import *

def convert_ycbcr_to_bgr(image, dataformat = 'nchw'):
    """Convert YCbCr format image to BGR format.
        # nchw -> nhwc
    """
    begin = time.time()

    r = 298.082 * image[2, :, :] / 256. + 408.583 * image[0, :, :] / 256. - 222.921
    g = 298.082 * image[2, :, :] / 256. - 100.291 * image[1, :, :] / 256. - 208.120 * image[0, :, :] / 256. + 135.576
    b = 298.082 * image[2, :, :] / 256. + 516.412 * image[1, :, :] / 256. - 276.836

    output = np.array([b, g, r]).astype(np.float32)# nchw
    output = output.transpose([1, 2, 0]) # nhwc
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
    output_py = convert_ycbcr_to_bgr(input_c) # nhwc

    output_c = np.fromfile(os.path.join(dir_path, 'Output_C_3'), dtype=np.float32) # nhwc
    output_c = output_c.reshape(H, W, C)
    output_c = output_c.flatten()

    compare_two_tensor(output_py.flatten(), output_c)
    #compare_two_tensor2(output_py.flatten(), output_c)
