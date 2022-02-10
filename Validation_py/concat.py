import torch
import os
import argparse
import time
import numpy as np
from compare import *

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
device = "cpu:0"

def concat_torch(input_c1, input_c2, output_c, N, C, H, W, flag = 0):
    print('execute pytorch concat'.format(flag))

    input_torch_tensor1 = torch.Tensor(input_c1).view(N, C, H, W).to(device)
    input_torch_tensor2 = torch.Tensor(input_c2).view(N, C, H, W).to(device)

    begin = time.time()

    output_py = torch.cat((input_torch_tensor1,input_torch_tensor2), flag)
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

    input_c1 = np.fromfile(os.path.join(dir_path, 'Input_C_c1'), dtype=np.float32)
    input_c2 = np.fromfile(os.path.join(dir_path, 'Input_C_c2'), dtype=np.float32)
    output_c = np.fromfile(os.path.join(dir_path, 'Output_C_c'), dtype=np.float32)

    concat_torch(input_c1, input_c2, output_c, N, C, H, W, flag)