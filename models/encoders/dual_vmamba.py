# 核心作用: 定义双流融合编码器。这是模型的特征提取核心，专门设计用来处理成对的PET和CT图像。
# 修改建议: 这是模型创新性的核心。如果您想设计新的多模态融合策略，例如改变融合模块（CRM, DCIM）或在不同阶段采用不同的融合方式，这里是主要修改点。

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from ..encoders.local_vmamba.region_mamba import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from utils.logger import get_logger
from ..encoders.vmamba import Backbone_VSSM, CrossMambaFusionBlock, ConcatMambaFusionBlock
from ..mamba_net_utils import ChannelRectifyModule
logger = get_logger()

#RGBXTransformer是核心类，它内部实例化了两个Backbone_VSSM（来自vmamba.py），分别作为PET和CT图像的特征提取主干 。
# 在编码器的每个阶段（共4个阶段），它会分别提取两种模态的特征。
class RGBXTransformer(nn.Module):
    def __init__(self,
                 num_classes=1000,      #类别数量
                 norm_layer=nn.LayerNorm,   #归一化层
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,     #含义: 绝对位置编码 (Absolute Position Embedding) 的开关
                 img_size=[512, 512],       #含义: 输入图像尺寸。定义了模型期望接收的图像的高度和宽度。这个参数对于计算位置编码的大小等是必需的。
                 patch_size=4,
                 drop_path_rate=0.2,        # 随机深度比率 (Drop Path Rate)。这是一种正则化技术，在训练时会以一定的概率随机“丢弃”整个残差连接，以防止模型过拟合。
                 **kwargs):
        super().__init__()

        self.ape = ape

        #33M左右
        # --- 1. 创建共享的Mamba主干网络VSSBlock ---
        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        # --- 2. 创建4个CRM模块 ---
        self.cross_mamba=nn.ModuleList(
            #dim=dims * (2 ** i): 这是在设置CRM模块要处理的特征图的通道数。当 i=0 (第1阶段)，通道数是 96 * (2**0) = 96。当 i=1 (第2阶段)，通道数是 96 * (2**1) = 192。
            #HW=(128*128/((2 ** i)*(2 ** i))): 这是在设置CRM模块要处理的特征图的空间尺寸 (H * W)。当 i=0 (第1阶段)，空间尺寸是 128*128。当 i=1 (第2阶段)，空间尺寸是 64*64。
            ChannelRectifyModule(dim=dims * (2 ** i),HW=(128*128/((2 ** i)*(2 ** i))),reduction=16)   #(128*128/((2 ** i)*(2 ** i)))
            for i in range(4)
            )

        # --- 3. 创建4个DCIM模块 ---
        # ChannelRectifyModule(CRM)和Region_global_Block(DCIM)等自定义模块来融合这两个分支的特征图，实现跨模态信息交互。Z
        self.channel_attn_mamba = nn.ModuleList(
            Region_global_Block(
                outer_dim=dims * (2 ** i), inner_dim=dims * (2 ** i)
                ,num_words=16,drop_path=0)for i in range(4)  #num_words:4*4 small region
        )

        # --- 4. 创建4个为DCIM服务的Stem模块 ---
        self.region_patch = nn.ModuleList(
            Stem(inner_dim=dims * (2 ** i),outer_dim=dims * (2 ** i))
            for i in range(4)
        )
        #绝对位置编码的作用：像Mamba和Transformer这样的模型，它们在处理图像时，是将图像看作一系列“补丁（Patch）”的序列。但这种序列化会丢失每个“补丁”在原始2D图像中的绝对位置信息（例如，“这个补丁在图像的左上角”，“那个补丁在中间”）。
        # absolute_pos_embed 就是一个可学习的参数矩阵，它包含了这些位置信息。在处理特征时，将这个位置编码加到特征图上，可以帮助模型更好地理解空间结构。
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            #含义: 初始化两个空列表，分别用来存放CT分支 (absolute_pos_embed) 和PET分支 (absolute_pos_embed_x) 在每个阶段的位置编码。
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            #含义: 开始一个循环，遍历编码器的所有阶段（len(depths) 通常是4）。
            for i_layer in range(len(depths)):
                #计算当前阶段 i_layer 的特征图分辨率。
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))                #含义: 计算当前阶段的特征维度（通道数）

                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))    #创建一个可学习的参数张量，它的形状与当前阶段的特征图完全匹配 [1, 通道数, 高, 宽]。
                trunc_normal_(absolute_pos_embed, std=.02)    #使用“截断正态分布”来对这个位置编码参数进行科学的初始化。
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))  #为另一个分支（PET）也创建一个同样规格的位置编码参数。
                trunc_normal_(absolute_pos_embed_x, std=.02)

                # 将为当前阶段创建好的两个位置编码参数，分别添加到对应的列表中进行存储。
                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x C x H x W  #ct
        x_e  #pet
        """

        #1获取“接口图纸”



        #获取当前批次的大小（Batch Size），即一次处理多少张图像。
        B = x_rgb.shape[0]
        #这个列表将用来收集并存储每个阶段融合后的最终特征图，以供后续的解码器使用。
        outs_fused = []

        # --- 步骤A: 特征提取 ---
        #self.vssm 会返回一个包含4个阶段特征图的列表 outs_rgb
        outs_rgb = self.vssm(x_rgb) # B x C x H x W
        outs_x = self.vssm(x_e) # B x C x H x W


        # --- 步骤B: 逐阶段融合 ---
        for i in range(4):

            # 检查是否启用了绝对位置编码（APE）。如果启用，就从之前创建的位置编码列表中取出对应阶段的位置编码 self.absolute_pos_embed[i]，并将其加到当前阶段的CT特征图 outs_rgb[i] 上。PET分支 out_x 同理。
            if self.ape:
                # this has been discarded
                out_rgb = self.absolute_pos_embed[i].to(outs_rgb[i].device) + outs_rgb[i]
                out_x = self.absolute_pos_embed_x[i].to(outs_x[i].device) + outs_x[i]
            else:
                out_rgb = outs_rgb[i]
                out_x = outs_x[i]


            #CRM
            CRM = True
            #DCIM
            DCIM = True
            # --- 步骤C: 调用融合模块 ---
            if CRM and DCIM:
                # 经过CRM
                cross_rgb, cross_x = self.cross_mamba[i](out_rgb,out_x)

                # 经过为DCIM服务的Stem
                cross_rgb,cross_x,(H_out, W_out), (H_in, W_in) = self.region_patch[i](cross_rgb,cross_x)
                # 经过DCIM
                dcim_output = self.channel_attn_mamba[i](cross_rgb.contiguous(), cross_x.contiguous(),H_out, W_out,H_in, W_in).permute(0, 3, 1, 2).contiguous()


                x_fuse = out_rgb+out_x+dcim_output
            elif not DCIM and CRM:
                out_rgb,  out_x = self.cross_mamba[i](out_rgb, out_x)
                x_fuse = (out_rgb + out_x)
            elif DCIM and not CRM:
                cross_rgb = out_rgb
                cross_x = out_x
                cross_rgb, cross_x, (H_out, W_out), (H_in, W_in) = self.region_patch[i](cross_rgb, cross_x)
                x_fuse = out_rgb+out_x+self.channel_attn_mamba[i](cross_rgb.contiguous(), cross_x.contiguous(),
                                                                    H_out, W_out, H_in, W_in).permute(0, 3, 1,2).contiguous()

            elif not DCIM and not CRM:
                x_fuse = (out_rgb + out_x)

            #将当前阶段生成的最终融合特征 x_fuse 添加到 outs_fused 列表中。
            outs_fused.append(x_fuse)
        return outs_fused

    def forward(self, x_rgb, x_e):
        print("\n" + "=" * 20 + " 开始侦察 Shape " + "=" * 20)
        out = self.forward_features(x_rgb, x_e)
        return out


#vssm_tiny, vssm_small, vssm_base是预设好的不同规模的模型配置，方便一键切换 。
class vssm_tiny(RGBXTransformer):
    #**kwargs: 一个常用的Python语法，表示这个函数可以接受任意数量的额外关键字参数，增加了灵活性。
    #fuse_cfg=None: 一个可选参数，用于传入融合相关的配置，但在这里没有被直接使用。
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(            #调用父类的初始方法
            depths=[2, 2, 9, 2],    #含义：这是一个传递给父类的参数。它定义了Mamba主干网络四个阶段（Stage）的深度，即每个阶段分别包含2个、2个、9个和2个 VSSBlock 。这个列表直接决定了模型的深度。
            dims=96,        #第一个阶段输出的通道数味96，后续阶段的通道数在此基础翻倍192，384，768）
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',        #参数指定了预训练权重文件的路径。
            mlp_ratio=0.0,      #设置VSSBlock中的MLP比例，但在这里被设置为0.0，表示不使用MLP。
            downsample_version='v1',        #下采样版本，这里被设置为'v1'，表示使用v1版本。
            drop_path_rate=0,   #这是一种正则化技术，在训练时会随机“丢弃”某些残差连接，以防止模型过拟合。0 表示不使用随机深度。
        )

class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )

class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )