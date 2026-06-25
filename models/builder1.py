# 核心作用: 模型组装器。它不定义具体网络层，而是根据配置文件，将定义好的编码器（Backbone）和解码器（Decoder Head）“粘合”成一个完整的端到端模型。


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight     #该函数用于初始化神经网络的权重。
from functools import partial

from utils.logger import get_logger

logger = get_logger()           #获取一个日志记录器的实例，后续可以用logger.info()来打印信息。



#作为整个分割模型的框架
class EncoderDecoder(nn.Module):
    #根据传入的cfg.backbone配置，动态地从models.encoders.dual_vmamba导入并实例化不同大小的编码器（如vssm_tiny）
    # ignore_index=255表示在计算损失时忽略像素值为255的标签，通常用于标记图像中的“未知”或“边界”区域。
    # norm_layer: 指定归一化层，默认为nn.BatchNorm2d
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):

       #“去调用我父类（nn.Module）的初始化方法 (__init__)，把它该做的事情全部先做完！
       # ”它确保了您的自定义模型（子类 EncoderDecoder）能够正确地继承并激活其父类 torch.nn.Module 所提供的所有核心功能，为后续添加自定义网络层和进行训练打下坚实的基础。
        super(EncoderDecoder, self).__init__()

        #初始化一个默认的通道数列表，代表编码器在不同阶段输出特征图的通道数。这个值会被后续选择的具体主干网络覆盖。
        self.channels = [64, 128, 320, 512]
        # 将传入的归一化层保存为类的属性。
        self.norm_layer = norm_layer

        #动态调用encoder
        if cfg.backbone == 'sigma_tiny':
            #logger.info(...): 打印日志，告知用户正在使用的主干网络。
            logger.info('Using backbone: sigma_tiny')
            #四个阶段输出的通道数
            self.channels = [96, 192, 384, 768]
            #: 动态导入。从encoders/dual_vmamba.py文件中导入vssm_tiny函数，并将其重命名为backbone。
            from .encoders.dual_vmamba import vssm_tiny as backbone
            #调用backbone函数（即vssm_tiny()）来创建编码器实例，并将其赋值给self.backbone。
            self.backbone = backbone()
        elif cfg.backbone == 'sigma_small':
            logger.info('Using backbone: sigma_small')
            self.channels = [96, 192, 384, 768]
            from .encoders.dual_vmamba import vssm_small as backbone
            self.backbone = backbone()
        elif cfg.backbone == 'Swin_transformer':
            logger.info('Using backbone: Swin-transfomer')
            self.channels = [96, 192, 384, 768]
            from .encoders.dual_swin import swin_tiny as backbone
            self.backbone = backbone()
        else:
            logger.info('Using backbone: sigma_base')
            self.channels = [128, 256, 512, 1024]
            from .encoders.dual_vmamba import vssm_base as backbone
            self.backbone = backbone()

        #初始化一个辅助分割头aux_head为None。辅助头是一种训练技巧，在编码器的中间层添加一个额外的、较浅的分割头来辅助梯度回传，但在这个配置中未使用。
        self.aux_head = None

        #decoder
        if cfg.decoder == 'SwinDecoder':
            logger.info('Using SwinV2 Decoder')
            from .decoders.SwinV2_Decoder import SwinDecoder
            #深度监督是一种训练技巧，它会在解码器的多个中间层都添加一个输出头来计算损失，以辅助梯度回传。
            self.deep_supervision = False
            #实例化MambaDecoder，并将其赋值给self.decode_head。传入了图像尺寸、编码器输出的通道数、类别数等参数。
            self.decode_head = SwinDecoder(
                # --- 基本參數 ---
                img_size=[cfg.image_height, cfg.image_width],
                patch_size=4,  # *** 添加：需要與 Swin 編碼器匹配 ***
                in_channels=self.channels,
                num_classes=cfg.num_classes,
                # --- 解碼器結構參數 (需要來自 cfg 或設為默認值) ---
                decoder_embed_dim=self.channels[0],  # *** 使用 encoder 的 embed_dim 作為 decoder 最淺層 dim ***
                decoder_depths=[2, 2, 2],  # *** 添加：示例值，建議從 cfg 讀取 ***
                decoder_num_heads=[6, 12, 24],  # *** 添加：示例值 (從淺到深)，建議從 cfg 讀取，需與 encoder 對應層匹配或自定義 ***
                window_size=8,  # *** 添加：需要與 Swin 編碼器匹配或自定義 ***
                # --- Swin Block 行為參數 (建議與 encoder 設置類似，或從 cfg 讀取) ---
                mlp_ratio=4.,  # *** 添加 ***
                qkv_bias=True,  # *** 添加 ***
                drop_rate=0.,  # *** 添加 ***
                attn_drop_rate=0.,  # *** 添加 ***
                drop_path_rate=0.1,  # *** 添加 ***
                norm_layer=norm_layer,  # *** 添加 ***
                use_checkpoint=False,  # *** 添加：根據需要設置 ***
                # --- 其他 ---
                deep_supervision=self.deep_supervision
            )

        else:
            logger.info('No decoder or unsupported decoder for Swin Transformer')
            self.decode_head = None  # 或者拋出錯誤

        self.criterion = criterion
        if self.criterion:  # 检查是否定义了损失函数。如果定义了损失函数（意味着模型用于训练），则调用init_weights方法来初始化权重。
            self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:  #检查是否提供了预训练模型路径
            if cfg.backbone != 'vmamba':
                logger.info('Loading pretrained model: {}'.format(pretrained))
                self.backbone.init_weights(pretrained=pretrained)           #调用主干网络自身的init_weights方法来加载预训练权重。
        logger.info('Initing weights ...')

        #调用从utils导入的init_weight函数，对解码器（self.decode_head）的权重进行初始化，这里使用的是kaiming_normal_方法，这是一种常用的权重初始化策略。
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    #这个方法定义了模型的核心数据流：从编码到解码。
    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        #如果不使用深度监督。
        if not self.deep_supervision:
            #获取输入图像的原始尺寸。
            orisize = rgb.shape
            # 将两路输入（rgb和modal_x）送入主干网络，得到多层特征图x。
            x = self.backbone(rgb, modal_x)
            # 将主干网络输出的特征x送入解码器，得到初步的分割结果out。
            out = self.decode_head.forward(x)
            #由于解码器输出的尺寸通常小于原始输入，使用双线性插值将out上采样到与输入图像相同的尺寸。
            out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
            #如果使用辅助分割头，则将主干网络输出的辅助特征送入辅助分割头，并使用双线性插值将其上采样到与输入图像相同的尺寸。
            if self.aux_head:
                aux_fm = self.aux_head(x[self.aux_index])
                aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
                return out, aux_fm
            #return out: 返回最终的分割图。
            return out
        else:
            #使用深度监督
            x = self.backbone(rgb, modal_x)
            x_last, x_output_0, x_output_1, x_output_2 = self.decode_head.forward(x)
            return x_last, x_output_0, x_output_1, x_output_2

    #这是nn.Module要求的标准前向传播方法。当你调用model(input)时，实际上就是执行这个forward方法。
    # rgb: 接收CT图像批次（通常被视为RGB图像中的R、G、B通道，即使它是灰度的）。modal_x: 接收另一个模态的图像批次，这里是PET图像。
    def forward(self, rgb, modal_x, label=None):
        if not self.deep_supervision:
            #检查是否存在一个“辅助头”（auxiliary head）。辅助头是一种训练技巧，在编码器的中间层增加一个额外的、较浅的分割头，用来提供额外的梯度信号，帮助模型训练。
            if self.aux_head:
                out, aux_fm = self.encode_decode(rgb, modal_x)      #并期望它返回两个输出：out (最终的分割结果) 和 aux_fm (来自辅助头的中间结果)
            else:
                out = self.encode_decode(rgb, modal_x)      #out = self.encode_decode(...): 就调用 self.encode_decode 方法，并只接收它返回的唯一一个输出，即最终的分割结果 out。

            return out
        else:
            x_last, x_output_0, x_output_1, x_output_2 = self.encode_decode(rgb, modal_x)

            return x_last

    #这是一个用于计算模型FLOPs（浮点运算次数）的辅助方法，用来衡量模型的计算复杂度。
    def flops(self, shape=(3, 480, 640)):
        from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
        import copy

        '''
        code from
        https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py#L4
        '''

        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
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
        return f"params {params} GFLOPs {sum(Gflops.values())}"



# print_jit_input_names，flops_selective_scan_fn，selective_scan_flop_jit这些是专门为flops方法服务的辅助函数。
# 它们的核心作用是为fvcore库提供计算Mamba核心操作selective_scan计算量的数学公式，因为fvcore默认不认识这个自定义操作。


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
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