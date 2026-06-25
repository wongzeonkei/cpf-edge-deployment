# 核心作用: 定义解码器网络。负责将编码器提取的深层、低分辨率的特征图逐步上采样，恢复到原始图像尺寸，并最终生成像素级的分割预测。
# MambaDecoder是主模块，它包含多个上采样层（layers_up）。
# 它接收来自编码器的多尺度特征作为输入。

import numpy as np
import torch.nn as nn
import torch
from torch.nn.modules import module
import torch.nn.functional as F
from ..encoders.vmamba import CVSSDecoderBlock
import torch.utils.checkpoint as checkpoint
from einops import rearrange


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2,
                 norm_layer=nn.LayerNorm):  # dim_scale:维度缩放因子。默认值为2，意味着这个模块的目标是将空间尺寸（高和宽）扩大2倍，同时将通道数减少一半。
        super().__init__()
        self.input_resolution = input_resolution  # 接收一个元组，代表输入特征图的空间分辨率（高 H, 宽 W）。
        self.dim = dim  # 输入特征图的通道数（C）。
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        # 这个归一化层将在上采样之后被调用。它的维度被设置为 dim // dim_scale（例如 768 // 2 = 384），正好是上采样后的新通道数。
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):  # x: 接收一个形状为 [批量大小, 高, 宽, 通道数] (channel-last) 的输入特征图。
        """
        x: B, H, W, C
        """

        x = self.expand(x)  # 形状变化: [B, H, W, C] -> [B, H, W, 2*C]。特征图的空间尺寸（H, W）不变，但通道数翻了一倍。
        B, H, W, C = x.shape
        # rearrange是einops库的张量重排函数：我们指定 p1=2, p2=2，那么 c 就等于 C / 4。因为 C 已经是 2*dim 了，所以 c 就等于 (2*dim) / 4 = dim / 2。这正好是我们想要的目标通道数。
        # 形状变化: [B, H, W, 2*C] -> [B, H*2, W*2, C/4]。也就是 [B, H*2, W*2, dim/2]
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)

        return x


class UpsampleExpand(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.linear = nn.Linear(dim, dim // 2, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        # contiguous()确保张量在内存中是连续存储的，这是某些后续操作（如 view）所必需的
        x = self.linear(x).permute(0, 3, 1, 2).contiguous()  # 形状变化: [B, H, W, C] -> [B, H, W, C/2]。空间尺寸（H, W）不变，但通道数减半。

        # 🌟 修复：在进行上采样之前强制转换为 float32，防止混合精度下的插值算子崩溃
        x = F.interpolate(x.float(), scale_factor=2, mode='bilinear', align_corners=False).to(x.dtype).permute(0, 2, 3,
                                                                                                               1).contiguous()

        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size * patch_size * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = self.expand(x)  # B, H, W, 16C
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size,
                      c=C // (self.patch_size ** 2))
        x = self.norm(x)
        return x


class FinalUpsample_X4(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        x = self.linear1(x).permute(0, 3, 1, 2).contiguous()  # B, C, H, W

        # 🌟 修复：强转 float32 保护插值
        x = F.interpolate(x.float(), scale_factor=2, mode='bilinear', align_corners=False).to(x.dtype).permute(0, 2, 3,
                                                                                                               1).contiguous()  # B, 2H, 2W, C

        x = self.linear2(x).permute(0, 3, 1, 2).contiguous()  # B, C, 2H, 2W

        # 🌟 修复：强转 float32 保护插值
        x = F.interpolate(x.float(), scale_factor=2, mode='bilinear', align_corners=False).to(x.dtype).permute(0, 2, 3,
                                                                                                               1).contiguous()  # B, 4H, 4W, C

        x = self.norm(x)
        return x


# Mamba_up是一个核心的上采样块，它内部使用CVSSDecoderBlock（一种Mamba模块）来处理特征，然后通过UpsampleExpand或PatchExpand进行上采样。
class Mamba_up(nn.Module):
    def __init__(self,
                 dim,  # 输入特征图的通道数
                 input_resolution,  # 输入特征图的空间分辨率（高 H, 宽 W）
                 depth,  # 当前层包含的Mamba模块的数量
                 dt_rank="auto",
                 d_state=4,
                 ssm_ratio=2.0,
                 attn_drop_rate=0.,
                 drop_rate=0.0,
                 mlp_ratio=4.0,
                 drop_path=0.1,
                 norm_layer=nn.LayerNorm,
                 upsample=None,  # 它接收一个上采样模块的类型
                 shared_ssm=False,
                 softmax_version=False,
                 use_checkpoint=False,
                 **kwargs):

        super().__init__()
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            CVSSDecoderBlock(  # 在每次循环中，都会创建一个 CVSSDecoderBlock 的实例。
                hidden_dim=dim,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop=drop_rate,
            )
            for i in range(depth)]  # 这是一个循环，会执行 depth 次。
        )

        if upsample is not None:
            self.upsample = UpsampleExpand(input_resolution, dim=dim, patch_size=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class MambaDecoder(nn.Module):
    def __init__(self,
                 img_size=[480, 640],  # 輸入圖像的原始尺寸
                 in_channels=[96, 192, 384, 768],
                 num_classes=40,
                 dropout_ratio=0.1,
                 embed_dim=96,
                 align_corners=False,
                 patch_size=4,
                 depths=[4, 4, 4, 4],  # 一個列表，定義了解碼器每個階段內部包含多少個處理模塊
                 mlp_ratio=4.,
                 drop_rate=0.0,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 deep_supervision=False,  # 一個布爾值開關，決定是否啟用“深度監督”模式。
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)  # 獲取解碼器的總層數（階段數），通常是4。
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.deep_supervision = deep_supervision

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_up = nn.ModuleList()  # 创建一个 ModuleList 容器，用于存放所有主要的上采样和处理层。
        for i_layer in range(self.num_layers):  # 開始一個循環，為解碼器的每一個階段（從深到淺）創建一個上採樣層。
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer)
            else:
                layer_up = Mamba_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                    input_resolution=(
                                        self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                        self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                    depth=depths[(self.num_layers - 1 - i_layer)],
                                    mlp_ratio=self.mlp_ratio,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                    use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)

        self.norm_up = norm_layer(embed_dim)
        if self.deep_supervision:
            self.norm_ds = nn.ModuleList([norm_layer(embed_dim * 2 ** (self.num_layers - 2 - i_layer)) for i_layer in
                                          range(self.num_layers - 1)])
            self.output_ds = nn.ModuleList([nn.Conv2d(in_channels=embed_dim * 2 ** (self.num_layers - 2 - i_layer),
                                                      out_channels=self.num_classes, kernel_size=1, bias=False) for
                                            i_layer in range(self.num_layers - 1)])

        self.up = FinalUpsample_X4(input_resolution=(img_size[0] // patch_size, img_size[1] // patch_size),
                                   patch_size=4, dim=embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

    def forward_up_features(self, inputs):  # B, C, H, W
        if not self.deep_supervision:
            for inx, layer_up in enumerate(self.layers_up):
                if inx == 0:
                    x = inputs[3 - inx]  # B, 768, 15, 20
                    x = x.permute(0, 2, 3, 1).contiguous()  # B, 15, 20, 768
                    y = layer_up(x)  # B, 30, 40, 384
                else:
                    B, C, H, W = inputs[3 - inx].shape

                    # 🌟 修复：强转 float32 保护插值
                    y = F.interpolate(y.permute(0, 3, 1, 2).contiguous().float(), size=(H, W), mode='bilinear',
                                      align_corners=False).to(y.dtype).permute(0, 2, 3, 1).contiguous()

                    x = y + inputs[3 - inx].permute(0, 2, 3, 1).contiguous()
                    y = layer_up(x)
            x = self.norm_up(y)
            return x
        else:
            x_upsample = []
            for inx, layer_up in enumerate(self.layers_up):
                if inx == 0:
                    x = inputs[3 - inx]
                    x = x.permute(0, 2, 3, 1).contiguous()
                    y = layer_up(x)
                    x_upsample.append(self.norm_ds[inx](y))
                else:
                    x = y + inputs[3 - inx].permute(0, 2, 3, 1).contiguous()
                    y = layer_up(x)
                    if inx != self.num_layers - 1:
                        x_upsample.append((self.norm_ds[inx](y)))
            x = self.norm_up(y)
            return x, x_upsample

    def forward(self, inputs):
        if not self.deep_supervision:
            x = self.forward_up_features(inputs)
            x_last = self.up_x4(x, self.patch_size)
            return x_last
        else:
            x, x_upsample = self.forward_up_features(inputs)
            x_last = self.up_x4(x, self.patch_size)

            # 🌟 核心修复区：在 Deep Supervision 分支的 F.interpolate 中，强制添加 .float()，计算完后再转回原类型
            x_output_0 = self.output_ds[0](
                F.interpolate(x_upsample[0].permute(0, 3, 1, 2).contiguous().float(), scale_factor=16, mode='bilinear',
                              align_corners=False).to(x.dtype)
            )
            x_output_1 = self.output_ds[1](
                F.interpolate(x_upsample[1].permute(0, 3, 1, 2).contiguous().float(), scale_factor=8, mode='bilinear',
                              align_corners=False).to(x.dtype)
            )
            x_output_2 = self.output_ds[2](
                F.interpolate(x_upsample[2].permute(0, 3, 1, 2).contiguous().float(), scale_factor=4, mode='bilinear',
                              align_corners=False).to(x.dtype)
            )
            return x_last, x_output_0, x_output_1, x_output_2

    def up_x4(self, x, pz):
        B, H, W, C = x.shape
        x = self.up(x)
        x = x.view(B, pz * H, pz * W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.output(x)
        return x