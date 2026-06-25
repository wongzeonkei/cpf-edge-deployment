import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# --- 導入新的 Transformer 塊 ---
from .axialnet_prior_guided import AxialTransformerBlockPriorGuided, MlpLN


# --- PatchMerging 層 (使用 LayerNorm) ---
class PatchMerging(nn.Module):
    """ Patch Merging Layer. Input: B H W C, Output: B H/2 W/2 2*C """
    def __init__(self, dim, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        """ x: B, H, W, C """
        B, H, W, C = x.shape
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # 必须精确对齐 (B, H, W, C) 的最后两个维度 (W和C)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            H, W = H + (H % 2), W + (W % 2) # 手动更新

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x) # B H/2 W/2 2*C

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

# --- 用於 LayerNorm 的 Permute 輔助類 ---
class PermuteLayerNorm(nn.Module):
     def __init__(self, dim, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
          super().__init__()
          self.norm = norm_layer(dim)

     def forward(self, x): # Input: B, C, H, W
          x = x.permute(0, 2, 3, 1).contiguous() # B, H, W, C
          x = self.norm(x)
          x = x.permute(0, 3, 1, 2).contiguous() # B, C, H, W
          return x

# --- 最終的 AxialPriorEncoder ---
class AxialPriorEncoder(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 9, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), pet_feature_dim=64, use_checkpoint=False, fixed_tau=0.01):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2**(self.num_layers - 1)) # 768
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.img_size = img_size
        self.patch_size = patch_size
        self.max_res = img_size // patch_size # 最大特徵圖邊長 (128)

        # --- Stem: Conv stride 4 ---
        self.patch_embed = nn.Sequential(
             nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
             # 輸出 B, C=embed_dim, H/4, W/4
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # --- 構建 4 個 Stage ---
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # 隨機深度

        current_dim = embed_dim
        for i_layer in range(self.num_layers):
            stage_blocks = nn.ModuleList()
            for i in range(depths[i_layer]):
                 stage_blocks.append(
                     AxialTransformerBlockPriorGuided(
                         dim=current_dim,
                         num_heads=num_heads[i_layer],
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias,
                         drop=drop_rate,
                         attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]) + i],
                         norm_layer=self.norm_layer, # 統一使用 LN
                         act_layer=nn.GELU, # 統一使用 GELU
                         max_kernel_size=self.max_res, # 傳遞最大尺寸 128
                         pet_feature_dim=pet_feature_dim,
                         fixed_tau=fixed_tau
                      )
                 )

            # Stage 之間的 Downsample 層 (除了最後一層)
            if i_layer < self.num_layers - 1:
                downsampler = PatchMerging(dim=current_dim, norm_layer=self.norm_layer)
                current_dim = int(current_dim * 2) # PatchMerging 輸出 2*dim
            else:
                downsampler = nn.Identity()

            self.layers.append(nn.ModuleDict({
                'blocks': stage_blocks, # ModuleList
                'downsample': downsampler
            }))

        # 為每個 stage 的輸出添加 Norm 層 (用於 Skip Connection)
        self.out_norms = nn.ModuleList()
        for i_layer in range(self.num_layers):
             dim = int(embed_dim * 2**i_layer)
             self.out_norms.append(PermuteLayerNorm(dim, self.norm_layer))

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
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # 🚨 核心修改 1：将 f_pet_list 设为可选参数 (默认为 None)
    def forward(self, x_ct, f_pet_list=None):
        """
        Input:
            x_ct (B, C_in, H, W): CT 图像
            f_pet_list (List[torch.Tensor] | None): MPN 输出的列表。为 None 时执行无先验基线。
        Output:
            List[torch.Tensor]: 包含4个阶段输出的 [B, C, H', W'] 列表
        """
        x = self.patch_embed(x_ct) # (B, C_embed, H/4, W/4)
        x = self.pos_drop(x)
        output_features = []

        current_feature = x # B C H W
        for i_layer in range(self.num_layers):
            stage_dict = self.layers[i_layer] # 直接訪問 ModuleDict
            blocks = stage_dict['blocks']
            downsample = stage_dict['downsample']

            # 🚨 核心修改 2：安全地分发先验特征
            if f_pet_list is not None:
                f_pet_current = f_pet_list[i_layer].to(x_ct.device)
            else:
                f_pet_current = None

            # 將 current_feature 傳遞給 blocks
            for blk in blocks:
                 current_feature = blk(current_feature, f_pet_current)

            # 應用 Stage 輸出的 Norm
            norm_layer = self.out_norms[i_layer]
            normed_output = norm_layer(current_feature) # 輸出 B C H' W'
            output_features.append(normed_output) # 存儲 B C H' W'

            # 應用下采样 (除了最後一層)
            if i_layer < self.num_layers - 1:
                # PatchMerging 期望 B H W C
                current_feature_permuted = current_feature.permute(0, 2, 3, 1).contiguous()
                downsampled_output = downsample(current_feature_permuted) # 輸出 B H/2 W/2 C_out
                # Permute back to B C H W for next stage
                current_feature = downsampled_output.permute(0, 3, 1, 2).contiguous()

        return output_features