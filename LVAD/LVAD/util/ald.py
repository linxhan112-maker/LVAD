import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(kernel_size=5, sigma=0.8):
    """
    生成高斯核，用于初始化卷积核
    """
    ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / torch.sum(kernel)


class SpatialLowPassConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SpatialLowPassConv, self).__init__()
        self.kernel_size = kernel_size
        # 定义可学习的卷积核参数
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        # 利用高斯核初始化所有卷积核，使之具备低通特性
        gaussian = gaussian_kernel(kernel_size, sigma=1.0)  # [kernel_size, kernel_size]
        gaussian = gaussian.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, k, k]
        with torch.no_grad():
            for i in range(out_channels):
                for j in range(in_channels):
                    self.weight[i, j] = gaussian.clone()

    def forward(self, x):
        # 对卷积核进行 softmax 归一化，确保低通效果
        weight_reshaped = self.weight.view(self.weight.size(0), self.weight.size(1), -1)
        weight_exp = torch.exp(weight_reshaped)
        weight_sum = weight_exp.sum(dim=2, keepdim=True)
        low_pass_weight = weight_exp / weight_sum
        low_pass_weight = low_pass_weight.view_as(self.weight)
        # 利用归一化后的卷积核提取低频特征
        return F.conv2d(x, low_pass_weight, bias=None, padding=self.kernel_size // 2)


class AdaptiveFusion(nn.Module):
    def __init__(self, channels):
        super(AdaptiveFusion, self).__init__()
        # 全连接层用于计算自适应权重
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

    def forward(self, origin_features, low_freq_features):
        # 通过全局平均池化获得通道描述符
        origin_gap = F.adaptive_avg_pool2d(origin_features, 1).view(origin_features.size(0), -1)
        low_gap = F.adaptive_avg_pool2d(low_freq_features, 1).view(low_freq_features.size(0), -1)
        # 拼接描述符后计算融合权重
        descriptor = torch.cat([origin_gap, low_gap], dim=1)
        weights = self.fc(descriptor).view(origin_features.size(0), origin_features.size(1), 1, 1)
        # 利用权重融合低频特征和原始特征
        return origin_features + weights * low_freq_features


class AdaptiveLowPassDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(AdaptiveLowPassDownsampling, self).__init__()
        self.sconv = SpatialLowPassConv(in_channels, out_channels, kernel_size)
        self.fusion = AdaptiveFusion(out_channels)

    def forward(self, x):
        low_freq_features = self.sconv(x)
        # 通过自适应融合获得增强后的特征表示
        return self.fusion(x, low_freq_features)


# 示例：在 encoder 前加入 Adaptive Low-pass Downsampling 模块
if __name__ == '__main__':
    # 假设输入特征图形状为 [B, C, H, W]
    x = torch.randn(100, 256, 20, 40)
    downsample = AdaptiveLowPassDownsampling(in_channels=256, out_channels=256, kernel_size=5)
    enhanced_features = downsample(x)
    print("输出特征尺寸:", enhanced_features.shape)  # 输出：[2, 64, 32, 32]