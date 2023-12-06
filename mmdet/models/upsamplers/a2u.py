import math

import torch
from torch import nn
import torch.nn.functional as F
from mmcv.ops.carafe import carafe


class InterChannelUpRaw(nn.Module):
    '''
    s: upsample kernel size
    group: upsample kernel group
    share: False means U and V are channelwise. True means U and V are shared among channels
    '''

    def __init__(self, c):
        # def __init__(self, c):
        super(InterChannelUpRaw, self).__init__()
        self.inchannel = c
        # self.s = conf.MODEL.up_kernel_size
        # self.k = conf.MODEL.encode_kernel_size
        # self.share = conf.MODEL.share
        # self.group = conf.MODEL.downupsample_group
        self.s = 3
        self.k = 6
        self.share = True
        self.group = 1
        self.d = 1
        self.k_u = self.k
        self.padding = int((self.k - 1) / 2) if self.k % 2 == 0 else self.k // 2
        self.padding_u = int((self.k_u - 1) / 2) if self.k_u % 2 == 0 else self.k_u // 2
        self.bnu = nn.Sequential(
            nn.GroupNorm(num_channels=c, num_groups=32),
            nn.LeakyReLU()
        )
        self.bnv = nn.Sequential(
            nn.GroupNorm(num_channels=c, num_groups=32),
            nn.LeakyReLU()
        )
        self.bconv_UP = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c, self.d * (self.s * 2) ** 2 * self.group, 1, 1, padding=0, bias=True),
        )
        self.bconv_DP = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c, self.d * (self.s * 2) ** 2 * self.group, 1, 1, padding=0, bias=True),
        )
        self.bconv_U = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.k_u, self.k_u)),
            nn.Conv2d(c, self.d if self.share else self.d * c, 1, 1, padding=0, groups=1, bias=True),
        )
        self.bconv_V = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.k, self.k)),
            nn.Conv2d(c, self.d if self.share else self.d * c, 1, 1, padding=0, groups=1, bias=True),
        )

    def forward(self, x):
        n, c, h, w = x.size()
        p_u = self.bconv_UP(x).view(n, (self.s * 2) ** 2 * self.group, self.d, 1, 1)
        p_d = self.bconv_DP(x).view(n, (self.s * 2) ** 2 * self.group, self.d, 1, 1)
        u = self.bconv_U(x).view(n, -1, self.k_u, self.k_u)
        v = self.bconv_V(x).view(n, -1, self.k, self.k)

        out_u, out_d = [], []

        for i, (u1, v1, p_u1, p_d1) in enumerate(zip(u, v, p_u, p_d)):
            out1, out2 = [], []
            for j in range(self.d):
                if self.share:
                    out1.append(F.conv2d(x[i].unsqueeze(1), u1[j].view(1, 1, self.k_u, self.k_u), bias=None, stride=2,
                                         padding=self.padding_u).view(1, c, int(h / 2), int(w / 2)))  # n*c*h*w
                    out2.append(F.conv2d(x[i].unsqueeze(1), v1[j].view(1, 1, self.k, self.k), bias=None, stride=2,
                                         padding=self.padding).view(1, c, int(h / 2), int(w / 2)))  # n*c*h*w
                else:
                    out1.append(
                        F.conv2d(x[i].unsqueeze(0), u1[j * c:(j + 1) * c].view(c, 1, self.k_u, self.k_u), bias=None,
                                 stride=2, groups=c, padding=self.padding_u).view(1, c, int(h / 2),
                                                                                  int(w / 2)))  # n*c*h*w
                    out2.append(F.conv2d(x[i].unsqueeze(0), v1[j * c:(j + 1) * c].view(c, 1, self.k, self.k), bias=None,
                                         stride=2, groups=c, padding=self.padding).view(1, c, int(h / 2),
                                                                                        int(w / 2)))  # n*c*h*w
            out1 = self.bnu(torch.cat(out1, 0)).view(c, self.d, int(h / 2), int(w / 2))
            out2 = self.bnv(torch.cat(out2, 0)).view(c, self.d, int(h / 2), int(w / 2))

            out_u.append(F.conv2d((out1 * out2).sum(dim=0, keepdim=True) / math.sqrt(c),
                                  p_u1.view((self.s * 2) ** 2 * self.group, self.d, 1, 1), stride=1, padding=0))
            out_d.append(F.conv2d((out1 * out2).sum(dim=0, keepdim=True) / math.sqrt(c),
                                  p_d1.view((self.s * 2) ** 2 * self.group, self.d, 1, 1), stride=1, padding=0))

        out_u = torch.cat(out_u, 0)
        # out_d = torch.cat(out_d, 0)
        out_u = out_u.view(n, self.s ** 2 * 4 * self.group, int(h / 2), int(w / 2))
        out_u = F.pixel_shuffle(out_u, 2).view(n, self.s ** 2, self.group, h, w)
        out_u = F.softmax(torch.sigmoid(out_u), dim=1).view(n, -1, h, w) if self.s > 1 else torch.sigmoid(out_u).view(n,
                                                                                                                      -1,
                                                                                                                      h,
                                                                                                                      w)
        # out_d = out_d.view(n, self.s ** 2 * 4 * self.group, int(h / 2), int(w / 2)).view(n, self.s ** 2 * 4, self.group,
        #                                                                                  int(h / 2), int(w / 2))
        # out_d = F.softmax(out_d, dim=1).view(n, -1, int(h / 2), int(w / 2))

        # out_d = F.pixel_shuffle(out_d, 2)  # Added

        return out_u


class A2UUpRaw(nn.Module):
    def __init__(self, in_channels, up_kernel=3, scale=2):
        super(A2UUpRaw, self).__init__()
        self.up_kernel = up_kernel
        self.scale = scale
        self.kernel_gen = InterChannelUpRaw(in_channels)

    def forward(self, feature_en, feature_de):
        kernels = self.kernel_gen(feature_en)
        return carafe(feature_de, kernels, self.up_kernel, 1, self.scale)


class InterChannelUp(nn.Module):
    '''
    s: upsample kernel size
    group: upsample kernel group
    share: False means U and V are channelwise. True means U and V are shared among channels
    '''
    def __init__(self, c):
        super(InterChannelUp, self).__init__()
        self.inchannel = c
        # self.s = conf.MODEL.up_kernel_size
        # self.k = conf.MODEL.encode_kernel_size
        # self.share = conf.MODEL.share
        # self.group = conf.MODEL.downupsample_group
        self.s = 3
        self.k = 6
        self.share = True
        self.group = 1
        self.d = 1
        self.k_u = self.k
        self.padding = int((self.k - 1) / 2) if self.k % 2 == 0 else self.k // 2
        self.padding_u = int((self.k_u - 1) / 2) if self.k_u % 2 == 0 else self.k_u // 2
        self.bnu = nn.Sequential(
            nn.GroupNorm(num_channels=c, num_groups=32),
            nn.LeakyReLU()
        )
        self.bnv = nn.Sequential(
            nn.GroupNorm(num_channels=c, num_groups=32),
            nn.LeakyReLU()
        )
        self.bconv_UP = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c, self.d * (self.s * 2) ** 2 * self.group, 1, 1, padding=0, bias=True),
        )
        self.bconv_U = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.k_u, self.k_u)),
            nn.Conv2d(c, self.d if self.share else self.d * c, 1, 1, padding=0, groups=1, bias=True),
        )
        self.bconv_V = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.k, self.k)),
            nn.Conv2d(c, self.d if self.share else self.d * c, 1, 1, padding=0, groups=1, bias=True),
        )

    def forward(self, x):
        n, c, h, w = x.size()
        p_u = self.bconv_UP(x).view(n, (self.s * 2) ** 2 * self.group, self.d, 1, 1)
        u = self.bconv_U(x).view(n, -1, self.k_u, self.k_u)
        v = self.bconv_V(x).view(n, -1, self.k, self.k)

        def repeat(x: torch.FloatTensor):
            return x.squeeze(0).repeat(1, self.d, 1, 1)

        if self.share:
            x = torch.cat(list(map(repeat, torch.split(x.unsqueeze(2), 1, dim=0))), dim=1)
            out1 = F.conv2d(x, u.view(n * self.d, 1, self.k_u, self.k_u), bias=None, stride=2, groups=n * self.d,
                            padding=self.padding_u).transpose(0, 1)
            out2 = F.conv2d(x, v.view(n * self.d, 1, self.k, self.k), bias=None, stride=2, groups=n * self.d,
                            padding=self.padding).transpose(0, 1)
        else:
            x = torch.cat(list(map(repeat, torch.split(x.unsqueeze(1), 1, dim=0))), dim=1)
            out1 = F.conv2d(x, u.view(n * self.d * c, 1, self.k_u, self.k_u), bias=None, stride=2,
                            groups=n * self.d * c, padding=self.padding_u).view(n * self.d, c, int(h / 2), int(w / 2))
            out2 = F.conv2d(x, v.view(n * self.d * c, 1, self.k, self.k), bias=None, stride=2, groups=n * self.d * c,
                            padding=self.padding).view(n * self.d, c, int(h / 2), int(w / 2))
        out1 = self.bnu(out1).transpose(0, 1)
        out2 = self.bnv(out2).transpose(0, 1)
        out_u = F.conv2d((out1 * out2).sum(dim=0, keepdim=True) / math.sqrt(c),
                         p_u.view(n * 4 * self.s ** 2 * self.group, self.d, 1, 1), stride=1, groups=n, padding=0)
        out_u = out_u.view(n, self.s ** 2 * 4 * self.group, int(h / 2), int(w / 2))
        out_u = F.pixel_shuffle(out_u, 2).view(n, self.s ** 2, self.group, h, w)
        out_u = F.softmax(torch.sigmoid(out_u), dim=1).view(n, -1, h, w) if self.s > 1 else torch.sigmoid(out_u).view(n,
                                                                                                                      -1,
                                                                                                                      h,
                                                                                                                      w)

        return out_u


class A2UUp(nn.Module):
    def __init__(self, in_channels, up_kernel=3, scale=2):
        super(A2UUp, self).__init__()
        self.up_kernel = up_kernel
        self.scale = scale
        self.kernel_gen = InterChannelUp(in_channels)

    def forward(self, feature_en, feature_de):
        kernels = self.kernel_gen(feature_en)
        return carafe(feature_de, kernels, self.up_kernel, 1, self.scale)
