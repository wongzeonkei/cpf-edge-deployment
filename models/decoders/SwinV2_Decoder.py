# 文件: models/decoders/SwinV2_Decoder.py (再次修正)

import numpy as np
import torch.nn as nn
import torch
from torch.nn.modules import module
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple
# --- 確保 SwinTransformerBlock 導入路徑正確 ---
try:
    from ..encoders.swinV2_T import SwinTransformerBlock, Mlp
except ImportError:
    print("錯誤：無法導入 SwinTransformerBlock。請檢查導入路徑。")
    class SwinTransformerBlock(nn.Module): # 佔位符
        def __init__(self, *args, **kwargs):
            super().__init__()
            dim = kwargs.get('dim', 96)
            self.dummy_layer = nn.Linear(dim, dim)
        def forward(self, x):
            print("警告：正在使用虛擬 SwinTransformerBlock")
            return self.dummy_layer(x)

# --- Transformer 兼容的上採樣層 ---
class UpsampleConv(nn.Module):
    """ 使用 ConvTranspose2d 進行上採樣 """
    def __init__(self, in_channels, out_channels, norm_layer=nn.LayerNorm):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    # *** 修改：接收 H, W 作為參數 ***
    def forward(self, x, H, W): # 新增 H, W 參數
        # Input x expected in B L C format
        B, L, C = x.shape
        # H = W = int(L**0.5) # --- 移除假設方形的程式碼 ---
        if H * W != L:
             raise ValueError(f"Feature map size mismatch? L={L}, H={H}, W={W}")
        x = x.permute(0, 2, 1).reshape(B, C, H, W) # -> B C H W
        x = self.upsample(x) # -> B C_out H*2 W*2
        # Output in B C H W format
        return x

# --- 主要的 Swin 解碼器類 ---
class SwinDecoder(nn.Module):
    def __init__(self,
                 img_size=[512, 512],
                 patch_size=4,
                 in_channels=[96, 192, 384, 768],
                 num_classes=1,
                 decoder_embed_dim=96,
                 decoder_depths=[2, 2, 2],
                 decoder_num_heads=[12, 6, 3], # 注意：這裡頭數應該是 [3, 6, 12] 或類似（從深到淺增加）？ 或者是 [24, 12, 6] -> [12, 6, 3]? 需確認
                 window_size=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 deep_supervision=False,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(decoder_depths)
        self.deep_supervision = deep_supervision
        if self.deep_supervision:
             print("警告：深度監督在此 SwinDecoder 示例中未完全實現。")

        self.patch_size = patch_size
        self.embed_dim = decoder_embed_dim

        # --- 計算並儲存各階段分辨率 ---
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.encoder_resolutions = [] # 編碼器各階段輸出特徵圖的分辨率 (B C H W)
        self.decoder_feature_resolutions = [] # 解碼器各階段輸入給 Swin 塊的特徵圖的分辨率 (用於給 UpsampleConv 提供 H, W)
        current_res_h, current_res_w = self.patches_resolution
        for i in range(len(in_channels)):
            res = (current_res_h, current_res_w)
            self.encoder_resolutions.append(res)
            # 準備下一階段（如果不是最後一層）
            if i < len(in_channels) - 1:
                current_res_h //= 2
                current_res_w //= 2
        # self.encoder_resolutions = [(128, 128), (64, 64), (32, 32), (16, 16)] (假設 patch_size=4)

        # 儲存 deepest_stage_processing 的輸出分辨率 (也是第一個 UpsampleConv 的輸入 H,W)
        self.decoder_feature_resolutions.append(self.encoder_resolutions[-1]) # (16, 16)
        # 計算後續每個解碼階段輸入給 Swin 塊之前的 H, W (也就是 UpsampleConv 的輸入 H,W)
        temp_res = self.encoder_resolutions[-1]
        for i in range(self.num_layers): # num_layers 是解碼階段數
             # 當前 Swin 塊組的輸入分辨率（也是下一次 Upsample 的輸入 H,W）
             current_swin_input_res = (temp_res[0] * 2, temp_res[1] * 2)
             self.decoder_feature_resolutions.append(current_swin_input_res)
             temp_res = current_swin_input_res
        # self.decoder_feature_resolutions for depths=[2,2,2] would be [(16, 16), (32, 32), (64, 64), (128, 128)]
        # 索引 0 對應 deepest_stage_processing 輸出, 索引 1 對應第一個 stage_blocks 輸出...
        # 所以第 i 個 upsample_layer 需要的分辨率是 self.decoder_feature_resolutions[i]
        # 第 i 個 decoder_stages 輸入分辨率是 self.decoder_feature_resolutions[i+1]
        # --- 分辨率計算結束 ---


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))]

        self.decoder_stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()

        current_dim = in_channels[-1]

        self.deepest_stage_processing = nn.Sequential(
            norm_layer(current_dim),
        )

        for i_layer in range(self.num_layers): # i_layer = 0, 1, 2 (從深到淺)
            target_dim = in_channels[-(i_layer + 2)] # 跳躍連接的維度 (in_channels[2], in_channels[1], in_channels[0])
            upsampled_dim = current_dim // 2
            fused_dim = upsampled_dim + target_dim

            upsample = UpsampleConv(current_dim, upsampled_dim)
            self.upsample_layers.append(upsample)

            fusion_conv = nn.Sequential(
                nn.Conv2d(fused_dim, target_dim, kernel_size=1, bias=False),
            )
            self.fusion_convs.append(fusion_conv)

            # 當前解碼階段 Swin 塊組的輸入分辨率
            decoder_stage_resolution = self.decoder_feature_resolutions[i_layer + 1] # 使用預計算的分辨率

            stage_depth = decoder_depths[i_layer]
            stage_num_heads = decoder_num_heads[i_layer]
            stage_drop_path = dpr[sum(decoder_depths[:i_layer]):sum(decoder_depths[:i_layer + 1])]

            stage_blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=target_dim,
                    input_resolution=decoder_stage_resolution, # *** 使用預計算值 ***
                    num_heads=stage_num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=stage_drop_path[i] if isinstance(stage_drop_path, list) else stage_drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint
                )
                for i in range(stage_depth)])
            self.decoder_stages.append(nn.Sequential(*stage_blocks))

            current_dim = target_dim # 更新維度

        # --- 最終上採樣 ---
        # 輸入維度是 in_channels[0] (e.g., 96)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels[0], self.embed_dim, kernel_size=2, stride=2),
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2)
        )

        # --- 輸出層 ---
        self.output_norm = norm_layer(self.embed_dim)
        self.output_conv = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=1, bias=False)


    def forward_up_features(self, inputs):
        x = inputs[-1]
        B, C_deep, H_deep, W_deep = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C_deep)
        y = self.deepest_stage_processing(x) # y 初始為 B L3 C_deep

        for i in range(self.num_layers): # i = 0, 1, ..., num_layers-1 (從深到淺)
            upsample_layer = self.upsample_layers[i]
            fusion_conv = self.fusion_convs[i]
            decoder_block_group = self.decoder_stages[i]

            # *** 修改：獲取 y 的 H, W ***
            # 第 i 個 upsample_layer 輸入 y 的分辨率
            current_H, current_W = self.decoder_feature_resolutions[i]

            # 上採樣當前的 y (B L C -> B C_up H_up W_up)
            y_up = upsample_layer(y, current_H, current_W) # *** 傳遞 H, W ***

            skip = inputs[-(i + 2)] # 跳躍連接

            fused = torch.cat((y_up, skip), dim=1)
            adjusted = fusion_conv(fused)

            B_cur, C_target, H_up, W_up = adjusted.shape
            x = adjusted.permute(0, 2, 3, 1).reshape(B_cur, -1, C_target) # 準備 Swin 輸入

            y = decoder_block_group(x) # Swin 塊處理，輸出 B L C，作為下一次循環的 y

        return y # 返回 B L0 C0

    def forward(self, inputs):
        if self.deep_supervision:
            raise NotImplementedError("此 SwinDecoder 示例中未實現深度監督。")
        else:
            x = self.forward_up_features(inputs) # 輸出 B L0 C0

            B, L0, C0 = x.shape
            # *** 修改：獲取 H0, W0 ***
            # L0 對應的分辨率是 self.decoder_feature_resolutions 最後一個元素
            H0, W0 = self.decoder_feature_resolutions[-1] # L0 = H0 * W0
            # H0 = W0 = int(L0 ** 0.5) # --- 移除假設方形的程式碼 ---

            x_for_final_upsample = x.permute(0, 2, 1).reshape(B, C0, H0, W0) # -> B C0 H0 W0
            x = self.final_upsample(x_for_final_upsample) # 輸出 B C_final H_final W_final

            B, C_final, H_final, W_final = x.shape
            x = x.permute(0, 2, 3, 1) # -> B H_final W_final C_final
            x = self.output_norm(x)
            x = x.permute(0, 3, 1, 2) # -> B C_final H_final W_final (供 Conv2d 使用)

            x_last = self.output_conv(x) # B num_classes H_final W_final

            return x_last