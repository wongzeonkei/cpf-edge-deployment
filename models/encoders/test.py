# 文件路径: models/encoders/dual_vmamba.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from ..encoders.local_vmamba.region_mamba import Stem, Region_global_Block  # 保持 DCIM 相关导入
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from utils.logger import get_logger

# --- 修改点 1：导入 SwinTransformerV2 ---
# 假设 swin_transformer_v2.py 文件位于 models 目录下
try:
    from models.swin_transformer_v2 import SwinTransformerV2
except ImportError:
    # 如果在其他位置，您可能需要调整路径
    raise ImportError("无法导入 SwinTransformerV2，请检查路径")
# ------------------------------------
from ..mamba_net_utils import ChannelRectifyModule

logger = get_logger()


# --- 修改 RGBXTransformer 类 ---
class RGBXTransformer(nn.Module):
    def __init__(self,
                 num_classes=1,  # 通常用于分割，设为1
                 norm_layer=nn.LayerNorm,
                 # --- SwinV2-T 配置 ---
                 img_size=512,
                 patch_size=4,
                 in_chans=1,  # 输入通道为1
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=8,  # 使用 8x8 窗口
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,  # 参考 SwinV1-T
                 ape=False,  # SwinV2 通常不用绝对位置编码
                 patch_norm=True,
                 use_checkpoint=False,  # 根据需要设置
                 pretrained=None,  # 预训练权重路径
                 # ---------------------
                 **kwargs):
        super().__init__()
        self.ape = ape  # 虽然可能不用，但保留属性以防万一

        # --- 修改点 2：实例化 SwinTransformerV2 作为共享主干 ---
        # 注意：不再使用 Backbone_VSSM
        self.swin_v2_backbone = SwinTransformerV2(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # 设置为0，因为我们只用作特征提取器
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            # SwinV2 通常没有 downsample_version 参数
            # pretrained_window_sizes 通常用于 V2，这里先设为默认
            pretrained_window_sizes=[0, 0, 0, 0]
        )
        self.num_stages = len(depths)

        # --- 加载预训练权重 ---
        if pretrained:
            self.swin_v2_backbone.load_pretrained(pretrained)  # 调用SwinV2自带的加载方法

        # --- CRM, DCIM, Stem 保持不变 (它们的维度基于 embed_dim 计算) ---
        # 计算每个阶段的维度
        dims_list = [int(embed_dim * (2 ** i)) for i in range(self.num_stages)]

        # 创建CRM模块
        self.cross_mamba = nn.ModuleList([
            ChannelRectifyModule(
                dim=dims_list[i],
                # HW需要根据img_size, patch_size和阶段i计算
                HW=(img_size // patch_size // (2 ** i)) ** 2,  # 假设是方形
                reduction=16
            ) for i in range(self.num_stages)
        ])

        # 创建DCIM模块
        self.channel_attn_mamba = nn.ModuleList([
            Region_global_Block(
                outer_dim=dims_list[i],
                inner_dim=dims_list[i],
                num_words=16,  # 假设是4x4的小区域
                drop_path=0,  # 根据需要调整
                norm_layer=norm_layer
            ) for i in range(self.num_stages)
        ])

        # 创建为DCIM服务的Stem模块
        self.region_patch = nn.ModuleList([
            Stem(
                inner_dim=dims_list[i],
                outer_dim=dims_list[i]
            ) for i in range(self.num_stages)
        ])
        # -----------------------------------------------------

        # --- (可选) 绝对位置编码 ---
        # 如果您的 SwinTransformerV2 实现需要 APE，则保留，否则可以删除
        if self.ape:
            # SwinV2的APE可能与VSSM不同，需要适配
            # 这里的代码需要根据 SwinTransformerV2 的具体实现来调整
            # ... (可能需要重写这部分代码以匹配SwinV2) ...
            pass  # 暂时跳过APE部分

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x C x H x W  #ct
        x_e  #pet
        """
        print("\n" + "=" * 20 + " 开始 DEBUG SwinV2 编码器 " + "=" * 20)
        B = x_rgb.shape[0]
        outs_fused = []

        # --- 修改点 3：从 SwinTransformerV2 提取多阶段特征 ---
        # SwinTransformerV2 通常通过其 layers 属性输出
        # 我们需要分别对 CT 和 PET 应用整个流程，并收集中间结果

        def extract_features(backbone, x):
            features = []
            x = backbone.patch_embed(x)  # B, L, C
            if backbone.ape:
                x = x + backbone.absolute_pos_embed  # 处理绝对位置编码（如果启用）
            x = backbone.pos_drop(x)

            for i, layer in enumerate(backbone.layers):
                # SwinTransformerV2 的 layer 输出的是处理后的 x (通常是 [B, L, C])
                # 我们需要在 downsample 之前获取特征

                # 获取当前阶段的空间分辨率 (H, W)
                # 注意: SwinV2 可能没有直接存储 H, W，需要从 L 推断或从 layer 内部获取
                H, W = backbone.layers[i].input_resolution

                # ---- 在 BasicLayer 内部获取特征 ----
                # SwinV2 的 BasicLayer forward 就是处理 x -> blocks -> downsample
                # 我们需要在 downsample 之前取 x

                processed_x = x  # 先假设 x 是进入 layer 前的状态
                for blk in layer.blocks:
                    processed_x = blk(processed_x)

                # 这是当前阶段的输出特征 (在 downsample 之前)
                # 需要 reshape 回 [B, H, W, C] 或 [B, C, H, W]
                current_feature = backbone.norm(processed_x)  # SwinV2 在 BasicLayer 后通常有 norm

                # --- 将 [B, L, C] 转回 [B, C, H, W] ---
                L_stage = H * W
                C_stage = current_feature.shape[-1]
                feature_map = current_feature.permute(0, 2, 1).reshape(B, C_stage, H, W)
                features.append(feature_map)
                # ------------------------------------

                # 应用 downsample 为下一阶段做准备（如果不是最后一层）
                if layer.downsample is not None:
                    x = layer.downsample(processed_x)  # downsample 输入的是处理后的 x
                else:
                    x = processed_x  # 最后一层没有 downsample

            return features

        outs_rgb = extract_features(self.swin_v2_backbone, x_rgb)
        outs_x = extract_features(self.swin_v2_backbone, x_e)

        print("--- SwinV2 主干已处理完毕，输出4个阶段的特征列表 ---")
        for i in range(self.num_stages):
            print(f"提取的 CT 特征 - 阶段 {i + 1} shape: {outs_rgb[i].shape}")
            print(f"提取的 PET 特征 - 阶段 {i + 1} shape: {outs_x[i].shape}")
        print("-" * 60)
        # -----------------------------------------------------

        # --- 步骤B: 逐阶段融合 (这部分逻辑保持不变) ---
        for i in range(self.num_stages):
            print(f"--- 进入融合流程: 阶段 {i + 1} ---")

            if self.ape:
                # 如果使用了APE，需要确保这里的加法操作维度正确
                # SwinV2 的 APE 可能需要不同的处理方式
                # out_rgb = ...
                # out_x = ...
                out_rgb = outs_rgb[i]  # 暂时忽略APE，或根据SwinV2实现修改
                out_x = outs_x[i]
            else:
                out_rgb = outs_rgb[i]
                out_x = outs_x[i]

            print(f"阶段 {i + 1} 输入融合前的 CT shape: {out_rgb.shape}")
            print(f"阶段 {i + 1} 输入融合前的 PET shape: {out_x.shape}")

            # CRM 和 DCIM 的调用逻辑保持不变
            CRM = True
            DCIM = True
            if CRM and DCIM:
                cross_rgb, cross_x = self.cross_mamba[i](out_rgb, out_x)
                print(f"经过 CRM 后的 CT 特征 shape: {cross_rgb.shape}")
                print(f"经过 CRM 后的 PET 特征 shape: {cross_x.shape}")

                temp_rgb, temp_x, (H_out, W_out), (H_in, W_in) = self.region_patch[i](cross_rgb, cross_x)

                dcim_output = self.channel_attn_mamba[i](temp_rgb.contiguous(), temp_x.contiguous(), H_out, W_out, H_in,
                                                         W_in).permute(0, 3, 1, 2).contiguous()
                print(f"DCIM 模块的输出 shape: {dcim_output.shape}")

                x_fuse = out_rgb + out_x + dcim_output

            elif not DCIM and CRM:
                out_rgb, out_x = self.cross_mamba[i](out_rgb, out_x)
                x_fuse = (out_rgb + out_x)
            elif DCIM and not CRM:
                cross_rgb = out_rgb
                cross_x = out_x
                temp_rgb, temp_x, (H_out, W_out), (H_in, W_in) = self.region_patch[i](cross_rgb, cross_x)
                dcim_output = self.channel_attn_mamba[i](temp_rgb.contiguous(), temp_x.contiguous(), H_out, W_out, H_in,
                                                         W_in).permute(0, 3, 1, 2).contiguous()
                x_fuse = out_rgb + out_x + dcim_output
            elif not DCIM and not CRM:
                x_fuse = (out_rgb + out_x)

            print(f"阶段 {i + 1} 最终融合 (x_fuse) 的 shape: {x_fuse.shape}\n")
            outs_fused.append(x_fuse)

        print("=" * 22 + " DEBUG 结束 " + "=" * 22 + "\n")
        return outs_fused

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out


# --- 便捷的创建函数 (可以保留或修改) ---
# 注意：vssm_tiny, small, base 这些名字可能不再适用
# 您可以重命名或删除它们，或者保留作为参考
class swinv2_tiny_ws8_encoder(RGBXTransformer):  # 重命名
    def __init__(self, **kwargs):
        super(swinv2_tiny_ws8_encoder, self).__init__(  # 调用修改后的RGBXTransformer初始化
            img_size=512,
            patch_size=4,
            in_chans=1,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            pretrained=kwargs.get('pretrained', None),  # 从kwargs获取预训练路径
            **kwargs
        )

# 您可以不再需要 vssm_small 和 vssm_base 的定义了