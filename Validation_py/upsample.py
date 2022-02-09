import torch
import os
import argparse
import time
import numpy as np
from compare import *

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
device = "cpu:0"

def upsample_torch(input_c, output_c, N, C, H, W, flag = 0):

    if flag == 0 :
        mode_ = 'nearest'
        upsample_torch_api = torch.nn.Upsample(scale_factor=2, mode=mode_,).to(device)
    elif flag == 1:
        mode_ = 'bilinear'
        upsample_torch_api = torch.nn.Upsample(scale_factor=2, mode=mode_, align_corners=True).to(device)
    elif flag == 2:
        mode_ = 'bicubic'
        upsample_torch_api = torch.nn.Upsample(scale_factor=2, mode=mode_, align_corners=True).to(device)

    print('execute pytorch {} interpolation!!!'.format(mode_))

    input_torch_tensor = torch.Tensor(input_c).view(N, C, H, W).to(device)



    begin = time.time()

    output_py = upsample_torch_api(input_torch_tensor)
    #torch.cuda.synchronize()

    dur = time.time() - begin

    output_py = output_py.detach().cpu().data.numpy().flatten()

    compare_two_tensor(output_py, output_c)
    print('dur time(pytorch) : %.3f [msec]'%(dur * 1000))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='add parameters')

    parser.add_argument('--N', type=int, default=1, help='batch size')
    parser.add_argument('--C', type=int, default=3, help='channel')
    parser.add_argument('--H', type=int, default=1080, help='height')
    parser.add_argument('--W', type=int, default=1920, help='width')
    parser.add_argument('--flag', type=int, default=0, help='flag')

    args = parser.parse_args()

    dir_path = os.path.dirname(__file__)

    N = args.N
    C = args.C
    H = args.H
    W = args.W
    flag = args.flag
    output_c = np.fromfile(os.path.join(dir_path, 'Output_C_un'), dtype=np.float32)
    input_c = np.fromfile(os.path.join(dir_path, 'Input_C_un'), dtype=np.float32)

    upsample_torch(input_c, output_c, N, C, H, W, flag)