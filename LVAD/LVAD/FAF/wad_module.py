import torch
import torch.nn as nn
# from DWT import DWT_2D
from DWT_layer import DWT_1D, IDWT_1D, DWT_2D_tiny, DWT_2D, IDWT_2D
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import pywt
import imageio

class wad_module_new(nn.Module):
    '''
    This module is used in directly connected networks.
    X --> output
    Args:
        wavename: Wavelet family
    '''
    def __init__(self, wavename='haar'):  # wavelist() or [‘haar’, ‘db’, ‘sym’, ‘coif’, ‘bior’, ‘rbio’, ‘dmey’]
        super(wad_module_new, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.softmax = nn.Softmax2d()

        self.lam = nn.Parameter(torch.tensor(0.1))

        #self.shrink = nn.Softshrink(lambd=0.0)  # 初始 lambd 设为 0，下面动态传参

    @staticmethod
    def get_module_name():
        return "wad"

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)

        # 2. 应用 Soft Thresholding (去噪)
        # 确保阈值非负
        threshold = torch.abs(self.lam)

        # 对高频分量进行“收缩” (去噪)
        LH_clean = torch.sign(LH) * torch.relu(torch.abs(LH) - threshold)
        HL_clean = torch.sign(HL) * torch.relu(torch.abs(HL) - threshold)

        output = LL

        x_high = self.softmax(torch.add(LH_clean, HL_clean))

        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)

        return output
