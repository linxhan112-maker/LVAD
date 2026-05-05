import os
import glob
import cv2
import numpy as np
import torch

import torch
import torch.nn.functional as F
import math

class GPURetinexProcessor:
    def __init__(self, mode='msr', sigmas=[15, 80, 250], eps=1e-6):
        self.mode = mode
        self.sigmas = sigmas
        self.eps = eps

    def _gaussian_kernel(self, sigma, device, dtype):
        # 根据 sigma 计算核大小（保证为奇数）
        kernel_size = int(6 * sigma) | 1
        # 生成一维坐标
        ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        # 计算高斯权重（1D）
        gauss = torch.exp(-0.5 * (ax / sigma)**2)
        gauss = gauss / gauss.sum()
        # 生成 2D 高斯核（外积）
        kernel_2d = torch.outer(gauss, gauss)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, k, k)
        return kernel_2d

    def single_scale_retinex_batch(self, imgs, sigma):
        """
        imgs: tensor, shape (B, H, W, C), dtype float32, 数值建议已归一化或直接原始值
        """
        device = imgs.device
        dtype = imgs.dtype
        # 将 imgs 转换为 (B, C, H, W)
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs + 1.0  # 防止 log(0)
        log_imgs = torch.log(imgs)
        # 获取高斯核，并对每个通道独立应用（groups = C）
        kernel = self._gaussian_kernel(sigma, device, dtype)
        C = imgs.shape[1]
        kernel = kernel.expand(C, 1, kernel.shape[2], kernel.shape[3])
        # 边缘填充：采用相同填充（padding = kernel_size//2）
        padding = kernel.shape[2] // 2
        blurred = F.conv2d(imgs, kernel, padding=padding, groups=C)
        log_blurred = torch.log(blurred + self.eps)
        retinex = log_imgs - log_blurred
        # 对每个样本独立归一化到 [0, 255]
        B = retinex.shape[0]
        retinex_norm = []
        for b in range(B):
            r = retinex[b]
            r_min = r.min()
            r_max = r.max()
            # 线性归一化
            r_norm = (r - r_min) / (r_max - r_min + self.eps) * 255.0
            retinex_norm.append(r_norm)
        retinex_norm = torch.stack(retinex_norm, dim=0)
        # 转换回 (B, H, W, C)
        retinex_norm = retinex_norm.permute(0, 2, 3, 1)
        # 若需要高精度，可保持 float32；若需要类似 CV_8UC3，则转换为 uint8
        return retinex_norm.round().to(torch.uint8)

    def multi_scale_retinex_batch(self, imgs):
        B, H, W, C = imgs.shape
        msr = torch.zeros((B, H, W, C), device=imgs.device, dtype=torch.float32)
        for sigma in self.sigmas:
            ssr = self.single_scale_retinex_batch(imgs, sigma).float()
            msr += ssr / len(self.sigmas)
        # 再次归一化
        retinex_norm = []
        for b in range(B):
            r = msr[b].permute(2, 0, 1)  # (C,H,W)
            r_min = r.min()
            r_max = r.max()
            r_norm = (r - r_min) / (r_max - r_min + self.eps) * 255.0
            retinex_norm.append(r_norm.permute(1, 2, 0))
        retinex_norm = torch.stack(retinex_norm, dim=0)
        return retinex_norm.round().to(torch.uint8)

    def __call__(self, imgs):
        """
        imgs: tensor, shape (B, H, W, C)，应在 GPU 上
        """
        if self.mode == 'ssr':
            return self.single_scale_retinex_batch(imgs, self.sigmas[0])
        elif self.mode == 'msr':
            return self.multi_scale_retinex_batch(imgs)
        else:
            raise ValueError("不支持的 mode，请选择 'ssr' 或 'msr'。")

# 假设你已经定义好了 GPURetinexProcessor 类，并放在当前脚本或模块中

def load_images_from_dir(directory, exts=('jpg', 'jpeg', 'png', 'tif')):
    """
    从目录中加载所有图像，并返回列表（所有图像尺寸一致）
    """
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(directory, f'*.{ext}')))
        image_paths.extend(glob.glob(os.path.join(directory, f'*.{ext.upper()}')))
    imgs = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        # 为了批量处理需要尺寸一致，可选 resize（例如 256x256）
        img = cv2.resize(img, (640, 320))
        imgs.append(img)
    return imgs


def test_gpu_retinex_processor():
    # 设置测试图片所在的目录
    input_dir = r'C:\Users\hony\Desktop\lowlight'  # 这里请确保目录中有一些测试图像
    output_dir = r'C:\Users\hony\Desktop\lowlight-test'
    os.makedirs(output_dir, exist_ok=True)

    # 加载图像并转换为 batch numpy array，形状：(B, H, W, C)
    imgs = load_images_from_dir(input_dir)
    if len(imgs) == 0:
        print("没有加载到任何图像，请检查目录！")
        return

    batch_np = np.stack(imgs, axis=0)  # (B, H, W, C)

    # 转换为 torch tensor，确保数据类型为 float32
    # 如果原始图像是 0-255 的 uint8，转换为 float32 后仍保持范围，后续计算会加1防止 log(0)
    batch_tensor = torch.from_numpy(batch_np).to(torch.float32).to('cuda')

    # 创建 GPU Retinex 处理器实例
    processor = GPURetinexProcessor(mode='msr', sigmas=[15, 80, 250])

    # 禁用梯度计算，提高性能
    with torch.no_grad():
        # 调用处理器
        output_tensor = processor(batch_tensor)  # 输出形状仍为 (B, H, W, C)

    # 将结果转换回 CPU 的 numpy 数组
    output_np = output_tensor.cpu().numpy()

    # 保存结果
    for idx, out_img in enumerate(output_np):
        out_path = os.path.join(output_dir, f"output_{idx}.jpg")
        cv2.imwrite(out_path, out_img)
        print(f"保存处理后图像：{out_path}")


if __name__ == "__main__":
    test_gpu_retinex_processor()