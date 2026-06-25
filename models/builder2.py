# 核心作用: 模型组装器。它不定义具体网络层，而是根据配置文件，将定义好的编码器（Backbone）和解码器（Decoder Head）“粘合”成一个完整的端到端模型。

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.init_func import init_weight
from functools import partial
from utils.logger import get_logger

# --- 新增導入 ---
from .encoders.mpn import MetabolicPriorNetwork
from .encoders.axial_prior_encoder import AxialPriorEncoder
from .decoders.MambaDecoder import MambaDecoder
# --- 結束新增導入 ---

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer
        self.criterion = criterion

        # ==========================================
        # 🌟 消融实验核心参数获取 🌟
        # ==========================================
        self.ablation_mode = cfg.ablation_mode if hasattr(cfg, 'ablation_mode') else 4
        logger.info(f"==> Builder Initialized in Ablation Mode: {self.ablation_mode}")

        # --- 替換 CIPA Backbone ---
        img_size_h = cfg.image_height if hasattr(cfg, 'image_height') else 512
        img_size_w = cfg.image_width if hasattr(cfg, 'image_width') else 512
        patch_size = cfg.patch_size if hasattr(cfg, 'patch_size') else 4
        embed_dim = 96
        depths = [2, 2, 9, 2]
        num_heads = [3, 6, 12, 24]
        self.channels = [embed_dim * (2 ** i) for i in range(4)]  # [96, 192, 384, 768]
        pet_feature_dim = 64
        encoder_norm_layer = partial(nn.LayerNorm, eps=1e-6)
        encoder_act_layer = nn.GELU
        drop_path_rate = 0.5
        fixed_tau = cfg.fixed_tau if hasattr(cfg, 'fixed_tau') else 0.01

        logger.info(f'Using AxialPriorEncoder with embed_dim={embed_dim}, depths={depths}, num_heads={num_heads}')
        logger.info(f'Using fixed_tau={fixed_tau}')
        logger.info(f'Using MetabolicPriorNetwork with pet_feature_dim={pet_feature_dim}')

        # ==========================================
        # 🌟 Mode 1: Early Concat 降维卷积 🌟
        # ==========================================
        if self.ablation_mode == 1:
            # CT(3) + PET(1) = 4通道 -> 降维回3通道，交给标准编码器
            self.early_fusion_conv = nn.Conv2d(4, 3, kernel_size=1, bias=False)
            logger.info("==> Built Early Fusion Conv (4 -> 3 channels) for Mode 1")

        self.mpn = MetabolicPriorNetwork(
            in_chans=1,
            depths=[1, 1, 1, 1],
            dims=[32, 64, 128, 256],
            patch_size=patch_size,
            norm_layer=encoder_norm_layer,
            act_layer=encoder_act_layer,
            pet_feature_dim=pet_feature_dim
        )

        self.axial_encoder = AxialPriorEncoder(
            img_size=img_size_h,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            norm_layer=encoder_norm_layer,
            pet_feature_dim=pet_feature_dim,
            drop_path_rate=drop_path_rate,
            fixed_tau=fixed_tau
        )

        self.aux_head = None

        # decoder
        if cfg.decoder == 'MambaDecoder':
            logger.info('Using Mamba Decoder')
            self.deep_supervision = True
            self.decode_head = MambaDecoder(
                img_size=[img_size_h, img_size_w],
                in_channels=self.channels,
                num_classes=cfg.num_classes,
                embed_dim=self.channels[0],
                depths=[4, 4, 4, 4],
                norm_layer=encoder_norm_layer,
                deep_supervision=self.deep_supervision
            )
        else:
            logger.error(f'Decoder {cfg.decoder} not supported with this setup.')
            raise NotImplementedError

        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        if hasattr(self.decode_head, 'apply'):
            init_weight(self.decode_head, nn.init.kaiming_normal_,
                        self.norm_layer if not isinstance(self.norm_layer, partial) else nn.BatchNorm2d,
                        cfg.bn_eps, cfg.bn_momentum,
                        mode='fan_in', nonlinearity='relu')

    # 🚨 核心修改 1：允許 modal_x (PET) 為 None
    def encode_decode(self, rgb, modal_x=None):
        """使用 AxialEncoder (由 MPN 引導) 編碼 CT，然後用 MambaDecoder 解碼"""
        orisize = rgb.shape

        # ==========================================
        # 🌟 消融实验数据流动态路由 (物理级切断) 🌟
        # ==========================================
        if self.ablation_mode == 0:
            # --- Mode 0: 纯 CT 基线 ---
            # 彻底阻断 MPN 计算，直接传入 None 给编码器
            f_pet_list = None
            ct_features_list = self.axial_encoder(rgb, f_pet_list)

        elif self.ablation_mode == 1:
            # --- Mode 1: CT+PET Early Concat 融合 ---
            if modal_x is None:
                raise ValueError("Ablation Mode 1 requires PET image (modal_x), but got None.")
            # 通道拼接: (B, 3, H, W) + (B, 1, H, W) -> (B, 4, H, W)
            fused_input = torch.cat([rgb, modal_x], dim=1)
            # 通过 1x1 卷积降维回 3 通道
            fused_input = self.early_fusion_conv(fused_input)
            # 忽略 MPN 先验，将融合后的图像送入主干
            ct_features_list = self.axial_encoder(fused_input, None)

        else:
            # --- Mode 2, 3, 4: 先验引导注意力 (主架构) ---
            if modal_x is None:
                raise ValueError(f"Ablation Mode {self.ablation_mode} requires PET image (modal_x), but got None.")
            # 1. MPN 提取 PET 先驗 (B, Cp, H', W') 列表
            f_pet_list = self.mpn(modal_x)
            # 2. Axial Encoder 提取 CT 特徵，同時接收 PET 先驗進行引導
            ct_features_list = self.axial_encoder(rgb, f_pet_list)
        # ==========================================

        # 3. MambaDecoder 解碼
        if not self.deep_supervision:
            out = self.decode_head.forward(ct_features_list)
            out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
            return out
        else:
            x_last, x_output_0, x_output_1, x_output_2 = self.decode_head.forward(ct_features_list)
            # --- 修复维度丢失问题 ---
            if x_last.dim() == 3:
                x_last = x_last.unsqueeze(1)
            if x_output_0.dim() == 3:
                x_output_0 = x_output_0.unsqueeze(1)
            if x_output_1.dim() == 3:
                x_output_1 = x_output_1.unsqueeze(1)
            if x_output_2.dim() == 3:
                x_output_2 = x_output_2.unsqueeze(1)

            x_last = F.interpolate(x_last, size=orisize[2:], mode='bilinear', align_corners=False)
            x_output_0 = F.interpolate(x_output_0, size=orisize[2:], mode='bilinear', align_corners=False)
            x_output_1 = F.interpolate(x_output_1, size=orisize[2:], mode='bilinear', align_corners=False)
            x_output_2 = F.interpolate(x_output_2, size=orisize[2:], mode='bilinear', align_corners=False)
            return x_last, x_output_0, x_output_1, x_output_2

    # 🚨 核心修改 2：同樣允許 modal_x 為 None
    def forward(self, rgb, modal_x=None, label=None):
        if not self.deep_supervision:
            return self.encode_decode(rgb, modal_x)
        else:
            return self.encode_decode(rgb, modal_x)

    def flops(self, shape=(3, 480, 640)):
        from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
        import copy

        supported_ops={
            "aten::silu": None,
            "aten::neg": None,
            "aten::exp": None,
            "aten::flip": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = (torch.randn((1, *shape), device=next(model.parameters()).device), torch.randn((1, *shape), device=next(model.parameters()).device))
        print(len(input))
        for i in input:
            print(i.shape)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=input, supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    assert not with_complex
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops

def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops