import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 局部细节分支（深度可分离卷积）
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),  # 深度卷积
            nn.Conv2d(channels, channels, kernel_size=1)  # 逐点卷积
        )

        # 全局光照分支
        self.global_gap = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(channels, channels)

        # 上下文增强分支（空洞卷积）
        self.context_conv = nn.Conv2d(channels, channels,
                                      kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        # 局部细节 [100,256,20,40]
        local_feat = self.local_conv(x)

        # 全局光照 [100,256]
        global_feat = self.global_gap(x).squeeze(-1).squeeze(-1)  # 压缩空间维度
        global_feat = self.global_fc(global_feat)

        # 上下文特征 [100,256,20,40]
        context_feat = self.context_conv(x)

        return local_feat, global_feat, context_feat


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 通道注意力
        self.channel_fc = nn.Linear(channels, channels)

        # 空间注意力（融合局部和上下文特征）
        self.spatial_conv = nn.Conv2d(2 * channels, 1, kernel_size=3, padding=1)

    def forward(self, local, global_vec, context):
        # 通道注意力 [100,256]->[100,256,1,1]
        channel_attn = torch.sigmoid(
            self.channel_fc(global_vec)
        ).unsqueeze(-1).unsqueeze(-1)  # 恢复空间维度

        # 空间注意力 [100,256+256=512,20,40]->[100,1,20,40]
        spatial_feat = torch.cat([local, context], dim=1)  # 通道拼接
        spatial_attn = torch.sigmoid(self.spatial_conv(spatial_feat))

        return channel_attn, spatial_attn


class NoiseSuppression(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 噪声估计（轻量级）
        self.noise_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x, enhanced_feat):
        # 噪声掩膜 [100,1,20,40]
        noise_mask = torch.sigmoid(self.noise_conv(x))

        # 动态抑制：抑制噪声区域的特征
        return (1 - noise_mask) * enhanced_feat + noise_mask * x


class LLFA(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.multiscale = MultiScaleExtractor(channels)
        self.cs_attn = ChannelSpatialAttention(channels)
        self.noise_suppress = NoiseSuppression(channels)

    def forward(self, x):
        # 多尺度特征提取
        local, global_vec, context = self.multiscale(x)  # 各分支输出

        # 通道-空间注意力
        c_attn, s_attn = self.cs_attn(local, global_vec, context)

        # 特征增强（残差连接）
        attn_feat = c_attn * s_attn * x + x  # [100,256,20,40]

        # 噪声抑制
        output = self.noise_suppress(x, attn_feat)

        return output


if __name__ == "__main__":
    # 验证输入输出尺寸
    input_tensor = torch.randn(100, 256, 20, 40)  # 示例输入
    llfa = LLFA(channels=256)
    output = llfa(input_tensor)

    print(f"输入尺寸: {input_tensor.shape}")
    print(f"输出尺寸: {output.shape}")  # 应保持 [100,256,20,40]