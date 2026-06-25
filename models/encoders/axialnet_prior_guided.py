import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

# --- 輔助函數 ---
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷積"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class qkv_transform(nn.Conv1d): # 來自 MedT utils
    """用於 qkv_transform 的 Conv1d"""
    pass

# --- MLP 層 (使用 LayerNorm 和 GELU) ---
class MlpLN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x): # 輸入: (B, ..., C)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# --- 修改後的軸向注意力模塊 (純注意力, 無 Norm) ---
class AxialAttentionPriorGuided(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, max_kernel_size=128,  # 使用 max_kernel_size
                 attn_drop=0., proj_drop=0., width=False, fixed_tau=0.01):
        super(AxialAttentionPriorGuided, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.max_kernel_size = max_kernel_size  # 最大相對位置編碼範圍
        self.width = width  # True: 寬度軸注意力

        # QKV 投影 (使用 Linear)
        # 遵循 MedT 的 Q/K (dim//2), V (dim) 拆分 -> 總共 2*dim
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 最終投影回 dim
        self.proj_drop = nn.Dropout(proj_drop)

        # --- 可學習的先驗強度標量 ---
        self.tau = nn.Parameter(torch.ones(1) * fixed_tau)  # 初始化为外部传入的固定/可学习 tau

        # 相對位置編碼 (基於 max_kernel_size)
        self.group_planes = dim // num_heads  # head_dim
        # MedT RPE 矩陣大小
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, self.max_kernel_size * 2 - 1),
                                     requires_grad=True)

        self.reset_parameters()

    def _get_relative_embeddings(self, actual_kernel_size, device):
        """ 動態計算索引並查找 RPE """
        query_index = torch.arange(actual_kernel_size, device=device).unsqueeze(0)
        key_index = torch.arange(actual_kernel_size, device=device).unsqueeze(1)

        # 偏移以使用大 self.relative 矩陣的中心部分
        relative_index = key_index - query_index + self.max_kernel_size - 1
        # 裁剪 (確保索引在範圍內)
        relative_index = relative_index.clamp(min=0, max=self.max_kernel_size * 2 - 2)
        flatten_index = relative_index.view(-1)

        # 查找 RPE
        all_embeddings = torch.index_select(self.relative, 1, flatten_index).view(
            self.group_planes * 2, actual_kernel_size, actual_kernel_size)

        # 拆分 Q, K, V 的 RPE
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)

        return q_embedding, k_embedding, v_embedding

    # 🚨 核心修改 1: 允许 M_meta 为可选参数 (默认为 None)
    def forward(self, x, M_meta=None):
        """
        Args:
            x (torch.Tensor): 輸入特徵 (已 permute 且歸一化)。
                              高度:(B*W, H', C), 寬度:(B*H, W', C)
            M_meta (torch.Tensor | None): 代謝先驗矩陣。若為 None，則不注入先驗。
        Returns:
            torch.Tensor: 注意力輸出 (B*W, H', C) 或 (B*H, W', C)
        """
        BW_or_BH, K_Size, C = x.shape  # K_Size 是 H' 或 W'

        # QKV 投影
        qkv = self.qkv(x)  # (B*W/H, K_Size, 2*C)

        # 拆分 Q, K, V (遵循 MedT 拆分)
        qkv_reshaped = qkv.reshape(BW_or_BH, K_Size, self.num_heads, self.group_planes * 2).permute(0, 2, 1,
                                                                                                    3)  # (B*W/H, G, K_Size, 2*C_head)
        q, k, v = torch.split(qkv_reshaped,
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=3)
        # q, k: (B*W/H, G, K_Size, C_head//2)
        # v: (B*W/H, G, K_Size, C_head)

        # 獲取動態 RPE
        q_embedding, k_embedding, v_embedding = self._get_relative_embeddings(K_Size, x.device)

        # --- MedT RPE 邏輯 ---
        # 內容項 (Content-content)
        qk_c = torch.einsum('bgic,bgjc->bgij', q, q) * self.scale  # (B*W/H, G, K_Size, K_Size)

        # 位置項 (Content-position / Position-content)
        qr = torch.einsum('bgic,cij->bgij', q, q_embedding)  # (B*W/H, G, K_Size, K_Size)
        kr = torch.einsum('bgic,cij->bgij', k, k_embedding).transpose(-2, -1)  # (B*W/H, G, K_Size, K_Size)

        similarity_before_softmax = (qk_c + qr + kr)  # 組合

        # 🚨 核心修改 2: 安全阻断逻辑 (Bypass)。只有当传入了先验时，才注入先验
        if M_meta is not None:
            B, Spatial_dim_other, G, _, _ = M_meta.shape  # G = num_heads
            assert G == self.num_heads
            current_bs_times_spatial = similarity_before_softmax.shape[0]  # B*W or B*H
            M_meta_reshaped = M_meta.reshape(current_bs_times_spatial, self.num_heads, K_Size, K_Size)
            similarity_before_softmax = similarity_before_softmax + self.tau * M_meta_reshaped

        attn = similarity_before_softmax.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # --- 內容 (sv) 和 位置 (sve) 輸出 ---
        sv = torch.einsum('bgij,bgjc->bgic', attn, v)  # (B*W/H, G, K_Size, C_head)
        sve = torch.einsum('bgij,cij->bgic', attn, v_embedding)  # (B*W/H, G, K_Size, C_head)

        # 組合 (在 MedT 中它們是 cat 後 bn 再 sum，這裡我們簡化為相加)
        x = sv + sve  # (B*W/H, G, K_Size, C_head)

        x = x.transpose(1, 2).reshape(BW_or_BH, K_Size, C)  # (B*W/H, K_Size, C)

        # 最終投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def reset_parameters(self):
        # 初始化 Linear
        trunc_normal_(self.qkv.weight, std=.02)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        trunc_normal_(self.proj.weight, std=.02)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

        # 初始化 RPE
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


# --- 新的 Transformer 塊 (選項 B) ---
class AxialTransformerBlockPriorGuided(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 max_kernel_size=128, pet_feature_dim=64, fixed_tau=0.01):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.pet_feature_dim = pet_feature_dim
        self.max_kernel_size = max_kernel_size

        # --- 高度注意力路徑 ---
        self.norm_h = norm_layer(dim)
        self.attn_h = AxialAttentionPriorGuided(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, max_kernel_size=max_kernel_size,
            attn_drop=attn_drop, proj_drop=drop, width=False, fixed_tau=fixed_tau
        )
        self.drop_path_h = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # --- 寬度注意力路徑 ---
        self.norm_w = norm_layer(dim)
        self.attn_w = AxialAttentionPriorGuided(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, max_kernel_size=max_kernel_size,
            attn_drop=attn_drop, proj_drop=drop, width=True, fixed_tau=fixed_tau
        )
        self.drop_path_w = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # --- MLP 路徑 ---
        self.norm_mlp = norm_layer(dim)
        self.mlp = MlpLN(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path_mlp = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def calculate_prior(self, f_pet, H_prime, W_prime):
        """ 計算軸向分解代謝先驗矩陣 """
        B, C_p, H_p, W_p = f_pet.shape
        assert H_p == H_prime and W_p == W_prime, f"PET shape {f_pet.shape} != target {H_prime}x{W_prime}"
        assert C_p == self.pet_feature_dim, f"PET channel {C_p} != expected {self.pet_feature_dim}"
        num_heads = self.num_heads

        # --- 計算 M_meta_W (寬度先驗) ---
        f_pet_w = f_pet.permute(0, 2, 3, 1).contiguous().view(B * H_prime, W_prime, C_p)
        M_meta_W = torch.bmm(f_pet_w, f_pet_w.transpose(1, 2))
        M_meta_W = M_meta_W.view(B, H_prime, W_prime, W_prime).unsqueeze(2).expand(-1, -1, num_heads, -1,
                                                                                   -1)  # (B, H', G, W', W')

        # --- 計算 M_meta_H (高度先驗) ---
        f_pet_h = f_pet.permute(0, 3, 2, 1).contiguous().view(B * W_prime, H_prime, C_p)
        M_meta_H = torch.bmm(f_pet_h, f_pet_h.transpose(1, 2))
        M_meta_H = M_meta_H.view(B, W_prime, H_prime, H_prime).unsqueeze(2).expand(-1, -1, num_heads, -1,
                                                                                   -1)  # (B, W', G, H', H')

        scale = self.pet_feature_dim ** -0.5
        M_meta_W = M_meta_W * scale
        M_meta_H = M_meta_H * scale

        return M_meta_H, M_meta_W

    # 🚨 核心修改 3: 允许 f_pet 为可选参数 (默认为 None)
    def forward(self, x_ct, f_pet=None):
        """
        Args:
            x_ct (torch.Tensor): 輸入 CT 特徵 (B, C, H, W).
            f_pet (torch.Tensor | None): 輸入 PET 特徵. 若為 None，則不計算先驗。
        Returns:
            torch.Tensor: 輸出特徵 (B, C, H, W).
        """
        B, C, H_in, W_in = x_ct.shape

        # 🚨 核心修改 4: 安全获取先验矩阵，跳过 None
        if f_pet is not None:
            f_pet = f_pet.to(x_ct.device)
            M_meta_H, M_meta_W = self.calculate_prior(f_pet, H_in, W_in)
        else:
            M_meta_H, M_meta_W = None, None

        # --- 高度注意力 ---
        # 結構: x = x + Attn(LN(x))
        # 輸入 B C H W -> B H W C
        x_permuted = x_ct.permute(0, 2, 3, 1).contiguous()

        # 為高度軸準備: (B, H, W, C) -> (B, W, H, C) -> (B*W, H, C)
        x_h_in = x_permuted.transpose(1, 2).contiguous().view(B * W_in, H_in, C)
        x_norm_h = self.norm_h(x_h_in)
        attn_out_h = self.attn_h(x_norm_h, M_meta_H)  # 返回 (B*W, H, C)
        # Reshape back: (B, W, H, C) -> (B, H, W, C)
        attn_out_h_reshaped = attn_out_h.view(B, W_in, H_in, C).transpose(1, 2).contiguous()
        # Permute back to B C H W
        x_ct = x_ct + self.drop_path_h(attn_out_h_reshaped.permute(0, 3, 1, 2).contiguous())

        # --- 寬度注意力 ---
        # 結構: x = x + Attn(LN(x))
        # 輸入 B C H W -> B H W C
        x_permuted = x_ct.permute(0, 2, 3, 1).contiguous()

        # 為寬度軸準備: (B, H, W, C) -> (B*H, W, C)
        x_w_in = x_permuted.view(B * H_in, W_in, C)
        x_norm_w = self.norm_w(x_w_in)
        attn_out_w = self.attn_w(x_norm_w, M_meta_W)  # 返回 (B*H, W, C)
        # Reshape back: (B, H, W, C)
        attn_out_w_reshaped = attn_out_w.view(B, H_in, W_in, C)
        # Permute back to B C H W
        x_ct = x_ct + self.drop_path_w(attn_out_w_reshaped.permute(0, 3, 1, 2).contiguous())

        # --- MLP ---
        # 結構: x = x + MLP(LN(x))
        # 輸入 B C H W -> B H W C
        x_norm_mlp = self.norm_mlp(x_ct.permute(0, 2, 3, 1).contiguous())  # (B H W C)
        mlp_out = self.mlp(x_norm_mlp)  # (B H W C)
        # Permute back to B C H W
        x_ct = x_ct + self.drop_path_mlp(mlp_out.permute(0, 3, 1, 2).contiguous())

        return x_ct