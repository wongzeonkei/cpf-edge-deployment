# 文件路径: models/decoders/unet_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    """
    U-Net解码器中的一个基本块。
    包含：上采样 -> 拼接跳跃连接 -> 两个卷积层
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 上采样层，将特征图尺寸放大一倍
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 卷积层，处理拼接后的特征
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        # 拼接来自编码器的跳跃连接特征
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()

        # Swin-Tiny编码器输出的通道数是 [96, 192, 384, 768]
        # 我们需要反向构建解码器
        encoder_channels = encoder_channels[::-1] # 反转列表 -> [768, 384, 192, 96]

        self.center = nn.Sequential(
            nn.Conv2d(encoder_channels[0], 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        in_channels = [1024] + list(decoder_channels)
        skip_channels = list(encoder_channels[1:]) + [0] # 最后一层没有跳跃连接
        out_channels = decoder_channels

        self.blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            self.blocks.append(DecoderBlock(in_channels[i], skip_channels[i], out_channels[i]))

    def forward(self, encoder_features):
        # 反转编码器特征列表，从最深层开始处理
        encoder_features = encoder_features[::-1]

        # 最深层的特征
        x = self.center(encoder_features[0])

        # 逐层上采样
        for i, block in enumerate(self.blocks):
            skip = encoder_features[i + 1] if (i + 1) < len(encoder_features) else None
            x = block(x, skip)

        return x

class SegmentationHead(nn.Sequential):
    """最后的分割头，将解码器输出转换为最终的分割图"""
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=4):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class SwinUNet(nn.Module):
    """
    将Swin编码器和UNet解码器组合在一起
    """
    def __init__(self, num_classes=1):
        super().__init__()
        # Swin-Tiny编码器各阶段输出的通道数
        encoder_channels = [96, 192, 384, 768]
        # 自定义解码器各阶段的通道数
        decoder_channels = [256, 128, 64, 32]

        self.decoder = UNetDecoder(encoder_channels, decoder_channels)

        # 最终的分割头，将解码器的输出(32通道)变为最终的类别数(1)
        # 最终解码器输出的特征图尺寸是原图的1/4，所以需要上采样4倍
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1], # 32
            out_channels=num_classes,
            upsampling=4,
        )

    def forward(self, features):
        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        return masks