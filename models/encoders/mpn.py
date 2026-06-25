# models/encoders/mpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_
from functools import partial


# --- 輔助類：封裝 Permute + LayerNorm ---
class PermuteLayerNorm(nn.Module):
    def __init__(self, dim, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm = norm_layer(dim)

    def forward(self, x):  # Input: B, C, H, W
        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        return x


# --- 基礎卷積塊 (使用 LayerNorm 和 GELU) ---
class BasicConvBlockLN(nn.Module):
    def __init__(self, channels, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = PermuteLayerNorm(channels, norm_layer)  # 使用輔助類
        self.act = act_layer()

    def forward(self, x):  # Input: B, C, H, W
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# --- MPN 網絡 (重構後的版本) ---
class MetabolicPriorNetwork(nn.Module):
    def __init__(self, in_chans=3, depths=[1, 1, 1, 1], dims=[32, 64, 128, 256], patch_size=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, pet_feature_dim=64):
        """
        Args:
            in_chans (int): 輸入通道數 (3)
            depths (list): 每個階段的卷積塊數量 (例如 [1,1,1,1])
            dims (list): 每個階段的內部通道數 [32, 64, 128, 256]
            patch_size (int): 初始下采样因子 (4)
            norm_layer (nn.Module): 歸一化層 (LayerNorm)
            act_layer (nn.Module): 激活函數 (GELU)
            pet_feature_dim (int): 輸出 PET 特徵的通道維度 (64)
        """
        super().__init__()
        self.num_stages = len(depths)
        self.dims = dims
        self.pet_feature_dim = pet_feature_dim

        # --- Stem (拆分為 conv 和 norm_act) ---
        self.stem_conv = nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, padding=0, bias=False)
        if isinstance(norm_layer, partial) and norm_layer.func == nn.LayerNorm:
            self.stem_norm_act = nn.Sequential(PermuteLayerNorm(dims[0], norm_layer),
                                               act_layer() if act_layer is not None else nn.Identity())
        else:  # 備用 BatchNorm
            self.stem_norm_act = nn.Sequential(norm_layer(dims[0]),
                                               act_layer() if act_layer is not None else nn.Identity())
        # ------------------------------------

        self.stages = nn.ModuleList()
        in_dim = dims[0]  # Stem 的輸出維度
        for i_stage in range(self.num_stages):
            stage_layers = []
            out_dim = dims[i_stage]  # 當前 stage 的內部目標維度

            # --- 下采样層 (除第一個 stage 外) ---
            if i_stage > 0:
                # 輸入維度是上一個 stage 的輸出 pet_feature_dim
                downsample_conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, padding=0, bias=False)
                if isinstance(norm_layer, partial) and norm_layer.func == nn.LayerNorm:
                    downsample_norm_act = nn.Sequential(PermuteLayerNorm(out_dim, norm_layer),
                                                        act_layer() if act_layer is not None else nn.Identity())
                else:  # BatchNorm
                    downsample_norm_act = nn.Sequential(norm_layer(out_dim),
                                                        act_layer() if act_layer is not None else nn.Identity())
                stage_layers.append(nn.Sequential(downsample_conv, downsample_norm_act))
            # else:
            # 第一個 stage 的 in_dim (dims[0]) 和 out_dim (dims[0]) 相同, 無需額外處理

            # --- 基礎卷積塊 ---
            for _ in range(depths[i_stage]):
                stage_layers.append(BasicConvBlockLN(out_dim, norm_layer=norm_layer, act_layer=act_layer))

            # --- 最終 1x1 投影到 pet_feature_dim ---
            final_proj = nn.Conv2d(out_dim, pet_feature_dim, kernel_size=1, stride=1, padding=0)
            if isinstance(norm_layer, partial) and norm_layer.func == nn.LayerNorm:
                final_norm = PermuteLayerNorm(pet_feature_dim, norm_layer)
            else:  # BatchNorm
                final_norm = norm_layer(pet_feature_dim)
            stage_layers.append(nn.Sequential(final_proj, final_norm))

            self.stages.append(nn.Sequential(*stage_layers))
            in_dim = pet_feature_dim  # 更新下一個 stage 的輸入維度

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))  # 修正 fan_Fout -> fan_out
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_pet):
        """
        Input: x_pet (B, C_in, H, W)
        Output: List[torch.Tensor]: 包含4個階段輸出的 B C H W 特徵圖列表
        """

        # --- 匹配重構後的 __init__ ---
        x = self.stem_conv(x_pet)
        x = self.stem_norm_act(x)
        # ---------------------------

        feature_maps = []
        current_feature = x  # (B, dims[0], H/4, W/4)

        for i_stage in range(self.num_stages):
            stage_module = self.stages[i_stage]

            # 在重構後的 __init__ 中，stage_module 是一個完整的 Sequential
            output_feature = stage_module(current_feature)

            feature_maps.append(output_feature)  # (B, pet_feature_dim, H/..., W/...)

            # 更新 current_feature 以便下一個 stage 使用
            current_feature = output_feature

        return feature_maps