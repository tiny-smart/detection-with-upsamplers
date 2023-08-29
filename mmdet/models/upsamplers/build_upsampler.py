import torch.nn as nn
from mmcv.ops.carafe import CARAFEPack as CARAFE
from .fade import FADE
from .sapa_base import SAPA
from .sapa_deform import SAPADeform
from .sierra import SIERRA
from .dysample import DySample


def build_upsampler(cfg, in_channels, scale_factor):
    upsample_mode = cfg['type']
    if upsample_mode == 'nearest':
        return nn.Upsample(scale_factor=scale_factor, mode='nearest')
    elif upsample_mode == 'bilinear':
        bilinear_cfg = dict(align_corners=False)
        bilinear_cfg.update(cfg)
        return nn.Upsample(scale_factor=scale_factor,
                           mode='bilinear',
                           align_corners=bilinear_cfg['align_corners'])
    elif upsample_mode == 'deconv':
        deconv_cfg = dict(kernel_size=3,
                          stride=2,
                          padding=1,
                          output_padding=1)
        deconv_cfg.update(cfg)
        return nn.ConvTranspose2d(in_channels,
                                  in_channels,
                                  kernel_size=deconv_cfg['kernel_size'],
                                  stride=deconv_cfg['stride'],
                                  padding=deconv_cfg['padding'],
                                  output_padding=deconv_cfg['output_padding'])
    elif upsample_mode == 'pixelshuffle':
        pixelshuffle_cfg = dict(kernel_size=3,
                                padding=1)
        pixelshuffle_cfg.update(cfg)
        return nn.Sequential(nn.Conv2d(in_channels,
                                       in_channels * scale_factor ** 2,
                                       kernel_size=pixelshuffle_cfg['kernel_size'],
                                       padding=pixelshuffle_cfg['padding']),
                             nn.PixelShuffle(upscale_factor=scale_factor))
    elif upsample_mode == 'carafe':
        carafe_cfg = dict(up_kernel=5,
                          up_group=1,
                          encoder_kernel=3,
                          encoder_dilation=1,
                          compressed_channels=64)
        carafe_cfg.update(cfg)
        return CARAFE(channels=in_channels,
                      scale_factor=scale_factor,
                      up_kernel=carafe_cfg['up_kernel'],
                      up_group=carafe_cfg['up_group'],
                      encoder_kernel=carafe_cfg['encoder_kernel'],
                      encoder_dilation=carafe_cfg['encoder_dilation'],
                      compressed_channels=carafe_cfg['compressed_channels'])
    elif upsample_mode == 'fade':
        fade_cfg = dict(up_kernel_size=5,
                        gating=False)
        fade_cfg.update(cfg)
        return FADE(in_channels,
                    scale_factor=scale_factor,
                    up_kernel_size=fade_cfg['up_kernel_size'],
                    gating=fade_cfg['gating'])
    elif upsample_mode == 'sapa':
        sapa_cfg = dict(up_kernel_size=5,
                        embedding_dim=32,
                        qkv_bias=True,
                        norm=True)
        sapa_cfg.update(cfg)
        return SAPA(in_channels,
                    scale_factor=scale_factor,
                    up_kernel_size=sapa_cfg['up_kernel_size'],
                    embedding_dim=sapa_cfg['embedding_dim'],
                    qkv_bias=sapa_cfg['qkv_bias'],
                    norm=sapa_cfg['norm'])
    elif upsample_mode == 'sapadeform':
        sapadeform_cfg = dict(num_point=9,
                              groups=4,
                              embedding_dim=32,
                              qkv_bias=True,
                              high_offset=True,
                              norm=False)
        sapadeform_cfg.update(cfg)
        return SAPADeform(in_channels,
                          scale_factor=scale_factor,
                          num_point=sapadeform_cfg['num_point'],
                          groups=sapadeform_cfg['groups'],
                          embedding_dim=sapadeform_cfg['embedding_dim'],
                          qkv_bias=sapadeform_cfg['qkv_bias'],
                          high_offset=sapadeform_cfg['high_offset'],
                          norm=sapadeform_cfg['norm'])
    elif upsample_mode == 'sierra':
        return SIERRA(scale_factor=scale_factor)
    elif upsample_mode == 'dysample':
        dysample_cfg = dict(style='lp',
                            groups=4,
                            dyscope=False)
        dysample_cfg.update(cfg)
        return DySample(in_channels,
                        scale_factor=scale_factor,
                        style=dysample_cfg['style'],
                        groups=dysample_cfg['groups'],
                        dyscope=dysample_cfg['dyscope'])
    else:
        raise NotImplementedError
