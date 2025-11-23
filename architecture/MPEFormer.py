from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
import math
import warnings
from timm.models.layers import DropPath, to_2tuple
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.nn.modules.utils import _pair
from utils import My_summary

############################ truncated_normal初始化  ####################################
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
############################ truncated_normal初始化  ####################################

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Spatial Fusion Module
# replace original skip-connection
class SpatialFM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_gap_enc = nn.Conv2d(1, 1, 3, 1, 1)
        self.conv_gmp_enc = nn.Conv2d(1, 1, 3, 1, 1)
        self.conv_gap_dec = nn.Conv2d(1, 1, 3, 1, 1)
        self.conv_gmp_dec = nn.Conv2d(1, 1, 3, 1, 1)
        self.conv = nn.Conv2d(channel * 4, channel, 1, 1, bias=False)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x_enc, x_dec):
        # GAP: Global average pooling
        # GMP: Global mean pooling
        avg_pool_enc = torch.mean(x_enc, dim=1, keepdim=True)
        max_pool_enc, _ = torch.max(x_enc, dim=1, keepdim=True)
        avg_pool_dec = torch.mean(x_dec, dim=1, keepdim=True)
        max_pool_dec, _ = torch.max(x_dec, dim=1, keepdim=True)
        channel_attn_enc1 = self.conv_gap_enc(avg_pool_enc)
        channel_attn_enc2 = self.conv_gmp_enc(max_pool_enc)
        channel_attn_dec1 = self.conv_gap_dec(avg_pool_dec)
        channel_attn_dec2 = self.conv_gmp_dec(max_pool_dec)
        channel_attn_enc = (channel_attn_enc1 + channel_attn_enc2) / 2.0
        channel_attn_dec = (channel_attn_dec1 + channel_attn_dec2) / 2.0
        scale_enc = torch.sigmoid(channel_attn_enc).expand_as(x_enc)
        scale_dec = torch.sigmoid(channel_attn_dec).expand_as(x_dec)
        x_enc_after = x_enc * scale_enc
        x_dec_after = x_dec * scale_dec
        out = self.conv(torch.cat([x_enc,x_dec,x_enc_after,x_dec_after],dim=1))
        # channel_attn = (channel_attn_enc_sum + channel_attn_dec_sum) / 2.0
        # scale = torch.sigmoid(channel_attn).expand_as(x_enc)
        # x_after_channel = x_enc * scale
        # out = self.relu(x_after_channel)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2)) # 此时变为 [b,c,h,w]
        return out.permute(0, 2, 3, 1) # 此时变为 [b,h,w,c]

class MPESA_spectral(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            window_size,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.to_q = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, 3, 1, 1,groups=dim, bias=False)
        )
        self.to_k = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        )
        self.to_v = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        )
        self.rescale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, x):
        '''
        :param x: [b,h,w,c]
        :return out: [b,h,w,c]

        nW = num_windows = num_h * num_w, 窗口数量
        Wh, Ww = window_size, 窗口大小
        n = Wh*Ww, 窗口本身
        c = head * dim, 多头自注意力的通道

        从attn的格式看attn的类型:
        q,k,v: [b,n,head,nW,dim], 顺序不一定, 但必然含有这些要素
        只要切了块, 那么nW维就必然保留
        (n, n): spatial-wise, 矩阵乘法乘掉dim, 窗口内部的自注意力 (按理说应该不适用于MSFA demosaicing)
        (dim, dim)：spectral-wise, 矩阵乘法乘掉n, 非局部光谱自注意力
        (nW, nW): spatial-wise, 矩阵乘法乘掉dim, 窗口之间的自注意力
        '''
        b, h, w, c = x.shape
        x = self.norm1(x) # x: [b,h,w,c]

        w_size = to_2tuple(self.window_size)
        assert  h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap spatial size must be divisible by the window size'

        # non-local spectral MSA
        x = x.permute(0, 3, 1, 2) # x: [b,c,h,w]
        q_inp = self.to_q(x) # [b,c,h,w]
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b c (num_h b0) (num_w b1) -> b c (num_h num_w) (b0 b1)',
                                           b0=w_size[0], b1=w_size[1]),
                      (q_inp.clone(), k_inp.clone(), v_inp.clone())) # q, k, v: [b,c,nW,n]
        q, k, v = map(lambda t: rearrange(t, 'b (h d) nW n -> b h d nW n', h=self.num_heads),
                      (q, k, v)) # q, k, v: [b,head,dim,nW,n]
        q, k, v = map(lambda t: t.permute(0, 3, 1, 2, 4), (q, k, v)) # q, k, v: [b,nW,head,dim,n]

        sim = einsum('b N h i n, b N h j n -> b N h i j', q, k)  # sim: [b,nW,head,dim,dim]
        attn = sim * self.rescale
        attn = attn.softmax(dim=-1) # attn: [b,nW,head,dim,dim]
        out = einsum('b N h i j, b N h j n -> b N h i n', attn, v) # out: [b,nW,head,dim,n]

        out = rearrange(out, 'b N h d n -> b (h d) N n') # out: [b,c,nW,n]
        out = rearrange(out, 'b c (num_h num_w) (b0 b1) -> b c (num_h b0) (num_w b1)',
                        num_h=h//w_size[0], num_w=w//w_size[1], b0=w_size[0], b1=w_size[1]) # out: [b,c,h,w]
        out = self.to_out(out).permute(0, 2, 3, 1) # out: [b,h,w,c]
        return out


class MPESA_spatial(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            window_size,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.window_size = window_size
        self.scale = self.dim_head ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)

        self.norm1 = nn.LayerNorm(dim)

        # pos embed
        # w_size = to_2tuple(self.window_size)
        # self.pos_emb = nn.Sequential(
        #     nn.Conv2d(w_size[0] * w_size[1], w_size[0] * w_size[1], kernel_size=5, stride=1, padding=2, groups=w_size[0] * w_size[1]),
        #     nn.Conv2d(w_size[0] * w_size[1], w_size[0] * w_size[1], kernel_size=1, stride=1, padding=0, groups=self.num_heads)
        # )

    def forward(self, x):
        '''
        :param x: [b,h,w,c]
        :return out: [b,h,w,c]

        nW = num_windows = num_h * num_w, 窗口数量
        Wh, Ww = window_size, 窗口大小
        n = Wh*Ww, 窗口本身

        从attn的格式看attn的类型:
        q,k,v: [b,n,head,nW,dim], 顺序不一定, 但必然含有这些要素
        只要切了块, 那么nW维就必然保留
        (n, n): spatial-wise, 矩阵乘法乘掉dim, 窗口内部的自注意力 (按理说应该不适用于MSFA demosaicing)
        (dim, dim)：spectral-wise, 矩阵乘法乘掉n, 非局部光谱自注意力
        (nW, nW): spatial-wise, 矩阵乘法乘掉dim, 窗口之间的自注意力
        '''
        b, h, w, c = x.shape
        x = self.norm1(x)  # x:[b,h,w,c]

        w_size = to_2tuple(self.window_size)
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap spatial size must be divisible by the window size'

        # cross window (cross MSFA pattern) MSA
        x_inp = rearrange(x, 'b (num_h b0) (num_w b1) c -> b (num_h num_w) (b0 b1) c',
                          b0=w_size[0], b1=w_size[1]) # x_inp:[b,nW,n,c]
        q_inp = self.to_q(x_inp)  # q_inp:[b,nW,n,c]
        k_inp = self.to_k(x_inp)  # k_inp:[b,nW,n,c]
        v_inp = self.to_v(x_inp)  # v_inp:[b,nW,n,c]
        #p_inp = v_inp.clone() # p_inp:[b,nW,n,c]

        # q, k, v, p = map(lambda t: t.permute(0, 2, 1, 3),
        #                  (q_inp.clone(), k_inp.clone(), v_inp.clone(), p_inp.clone()))  # q,k,v,p:[b,n,nW,c]
        # q, k, v, p = map(lambda t: rearrange(t, 'b n nW (h d) -> b n h nW d', h=self.num_heads),
        #                  (q, k, v, p))  # q,k,v,p:[b,n,head,nW,dim]
        # p = rearrange(p,'b n h (num_h num_w) d -> (b h d) n num_h num_w',
        #               num_h=h//w_size[0], num_w=w//w_size[1])  # p:[b*c,n,num_h,num_w], num_h*num_w = nW
        q, k, v = map(lambda t: t.permute(0, 2, 1, 3),
                         (q_inp.clone(), k_inp.clone(), v_inp.clone()))  # q,k,v,p:[b,n,nW,c]
        q, k, v = map(lambda t: rearrange(t, 'b n nW (h d) -> b n h nW d', h=self.num_heads),
                         (q, k, v))  # q,k,v:[b,n,head,nW,dim]

        q *= self.scale
        sim = einsum('b n h i d, b n h j d -> b n h i j', q, k)  # sim2:[b,n,head,nW,nW]
        attn = sim.softmax(dim=-1)  # attn: [b,n,head,nW,nW]

        # pos emb
        #pos_emb = self.pos_emb(p)  # pos_emb:[b*c,n,num_h,num_w]
        # pos_emb = rearrange(pos_emb, '(b h d) n num_h num_w -> b n h (num_h num_w) d',
        #                     h=self.num_heads, d=self.dim_head)  # pos_emb:[b*c,n,num_h,num_w]->[b,n,head,nW,dim]
        out = einsum('b n h i j, b n h j d -> b n h i d', attn, v) # out:[b,n,head,nW,dim]
        #out = out + pos_emb

        out = rearrange(out, 'b n h nW d -> b n nW (h d)')  # out:[b,n,nW,c]
        out = out.permute(0, 2, 1, 3)  # out:[b,nW,n,c]
        out = rearrange(out, 'b (num_h num_w) (b0 b1) c -> b (num_h b0) (num_w b1) c',
                        num_h=h//w_size[0], num_w=w//w_size[1], b0=w_size[0], b1=w_size[1])  # out:[b,h,w,c]
        out = self.to_out(out)  # out:[b,h,w,c]

        return out  # out:[b,h,w,c]


class DFSAB(nn.Module):
    def __init__(
            self,
            msfa_size,
            dim,
            stage,
            num_blocks,
            win_size,
    ):
        super().__init__()
        self.MPSM = MPSM(dim)
        self.blocks1 = nn.ModuleList([]) # spatial branch
        self.blocks2 = nn.ModuleList([]) # spectral branch
        for _ in range(num_blocks):
            self.blocks1.append(nn.ModuleList([
                MPESA_spatial(dim=dim, num_heads=2 ** stage, window_size=win_size),
                PreNorm(dim, FeedForward(dim=dim))  # LayerNorm
            ]))
        for _ in range(num_blocks):
            self.blocks2.append(nn.ModuleList([
                MPESA_spectral(dim=dim, num_heads=2 ** stage, window_size=win_size),
                PreNorm(dim, FeedForward(dim=dim)) # LayerNorm
            ]))
        self.fusion = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x_size = x.shape[2], x.shape[3] # h,w
        x_in = self.MPSM(x) + x # x_in:[b,c,h,w]
        x_in = x_in.permute(0, 2, 3, 1) # x_in:[b,h,w,c]

        x1 = x_in # x1:[b,h,w,c]
        for (spa_attn, ff) in self.blocks1:
            x1 = spa_attn(x1) + x1
            x1 = ff(x1) + x1
        x1 = x1.permute(0, 3, 1, 2) # x1:[b,c,h,w]

        x2 = x_in # x2:[b,h,w,c]
        for (spe_attn, ff) in self.blocks2:
            x2 = spe_attn(x2) + x2
            x2 = ff(x2) + x2
        x2 = x2.permute(0, 3, 1, 2) # x2:[b,c,h,w]

        out = x1 + x2 # out:[b,c,h,w]
        out = self.fusion(out)

        return out

class MPSM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv_pixel_wise = nn.Conv2d(self.dim, self.dim,1,1,0, bias=False) # pixel-wise gating
        self.avgpool_conv1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # downsample 1/2
            nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=False)
        ) # pixel-wise
        self.conv_in_one_msfa = nn.Conv2d(self.dim, self.dim,5,1,2, bias=False) # within one MSFA
        self.avgpool_conv2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # downsample 1/2
            nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False)
        ) # cross MSFAs

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True) # gating proj
        self.proj2 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True) # gating proj
        self.proj3 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True) # gating proj
        self.conv_cat = nn.Conv2d(self.dim*3, self.dim, 1, 1, 0, bias=False)

    def forward(self, x):
        '''
        :param x: [b,c,h,w]
        :return out: [b,c,h,w], identical size with sparse raw image
        '''
        b, c, h, w = x.size()
        # [b,c,h,w]->[b,c,1,1]->[b,c,1]->[b,1,c], [b,1,c]是conv1d处理的格式, 1是channel dimension
        g1 = self.proj1(self.pool(self.conv_pixel_wise(x)).squeeze(-1).transpose(-1,-2))
        g2 = self.proj2(self.pool(self.conv_pixel_wise(x)).squeeze(-1).transpose(-1,-2))
        g3 = self.proj3(self.pool(self.conv_pixel_wise(x)).squeeze(-1).transpose(-1,-2))
        # [b,1,c]->[b,c,1]->[b,c,1,1]
        g1 = g1.transpose(-1,-2).unsqueeze(-1)
        g2 = g2.transpose(-1,-2).unsqueeze(-1)
        g3 = g3.transpose(-1,-2).unsqueeze(-1)

        x1 = F.interpolate(self.avgpool_conv1(x), size=(h, w), mode='bilinear', align_corners=False) # x1:[b,c,h,w]
        x2 = self.conv_in_one_msfa(x) # x2:[b,c,h,w]
        x3 = F.interpolate(self.avgpool_conv2(x), size=(h,w), mode='bilinear', align_corners=False) # x3:[b,c,h,w]

        x1 *= g1
        x2 *= g2
        x3 *= g3

        # x1_g = self.pool(x1) * g1 # x1_g:[b,c,1,1]
        # x2_g = self.pool(x2) * g2 # x2_g:[b,c,1,1]
        # x3_g = self.pool(x3) * g3 # x3_g:[b,c,1,1]

        # x1 *= x1_g # x1:[b,c,h,w]
        # x2 *= x2_g # x2:[b,c,h,w]
        # x3 *= x3_g # x3:[b,c,h,w]

        out = torch.cat([x1, x2, x3], dim=1)
        out = self.conv_cat(out) + x

        return out

class SSFN(nn.Module):
    def __init__(self, msfa_size, num_blocks=[1, 1, 1]):
        super(SSFN, self).__init__()
        self.dim = msfa_size**2
        self.in_dim = msfa_size**2
        self.out_dim = msfa_size**2

        # input projection
        self.embedding = nn.Conv2d(self.in_dim, self.dim, 3, 1, 1, bias=False)

        '''Spatial
        '''
        # Encode Spatial
        dim_stage = self.dim
        self.Spatial_Block_0 = DFSAB(msfa_size=msfa_size, dim=dim_stage, stage=0, num_blocks=num_blocks[0], win_size=8)
        self.Spatial_down_0 = nn.Conv2d(dim_stage, dim_stage*2, 4, 2, 1, bias=False) # DownSample：HW减半，通道数翻倍
        dim_stage *= 2

        self.Spatial_Block_1 = DFSAB(msfa_size=msfa_size, dim=dim_stage, stage=1, num_blocks=num_blocks[1], win_size=4)
        self.Spatial_conv_0 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, bias=False)
        self.Spatial_down_1 = nn.Conv2d(dim_stage, dim_stage*2, 4, 2, 1, bias=False) # DownSample：HW减半，通道数翻倍
        dim_stage *= 2

        # Bottleneck
        self.Spatial_conv_1 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, bias=False)
        self.bottleneck_Spatial = DFSAB(msfa_size=msfa_size, dim=dim_stage, stage=2, num_blocks=num_blocks[-1], win_size=2)

        # Decoder Spatial
        self.Spatial_up_0 = nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.Spatial_Block_2 = DFSAB(msfa_size=msfa_size, dim=dim_stage // 2, stage=1, num_blocks=num_blocks[1], win_size=4)
        self.SpatialFM_0 = SpatialFM(dim_stage // 2)
        dim_stage = dim_stage // 2

        self.Spatial_up_1 = nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.Spatial_Block_3 = DFSAB(msfa_size=msfa_size, dim=dim_stage // 2, stage=0, num_blocks=num_blocks[0], win_size=8)
        self.SpatialFM_1 = SpatialFM(dim_stage // 2)

        # output projection
        self.mapping = nn.Conv2d(self.dim, self.out_dim, 3, 1, 1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x) # fea: [1,16,480,512]

        # '''Spectral
        # '''
        # Spectral Encoder
        x_0_0 = self.Spatial_Block_0(fea) # x_0_0: [1,16,480,512]
        fea = self.Spatial_down_0(x_0_0) # fea: [1,32,240,256]

        x_1_0 = self.Spatial_Block_1(fea) # x_1_0: [1,32,240,256]
        fea = self.Spatial_down_1(x_1_0) # fea: [1,64,120,128]

        # Bottleneck
        b_0 = self.bottleneck_Spatial(fea) # b_0: [1,64,120,128]

        # Spectral Decoder
        fea = self.Spatial_up_0(b_0) # fea: [1,32,240,256]
        fea = self.SpatialFM_0(x_1_0, fea) # fea: [1,32,240,256]
        x_2_0 = self.Spatial_Block_2(fea) # x_2_0: [1,32,240,256]

        fea = self.Spatial_up_1(x_2_0) # fea: [1,16,480,512]
        fea = self.SpatialFM_1(x_0_0, fea) # fea: [1,16,480,512]
        x_3_0 = self.Spatial_Block_3(fea) # x_3_0: [1,16,480,512]

        # Mapping
        out = self.mapping(x_3_0) + x # out: [1,16,480,512]

        return out

class PhaseEdgeExtraction(nn.Module):
    def __init__(self, num_features, use_dwconv=False):
        super(PhaseEdgeExtraction, self).__init__()
        if use_dwconv:
            self.PEE = nn.Sequential(
                nn.Conv2d(1, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, groups=num_features),  # 每个通道单独处理边缘
                # 可选：作为mask的时候再加
                # nn.Sigmoid()
            )
        else:
            self.PEE = nn.Sequential(
                nn.Conv2d(1, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                # 可选：作为mask的时候再加
                # nn.Sigmoid()
            )
        # self.rescale = nn.Parameter(torch.ones(num_features, 1, 1))

    @staticmethod
    def standardize(inp, eps=1e-8):
        mean = inp.mean(dim=(2, 3), keepdim=True)
        std = inp.std(dim=(2, 3), keepdim=True)
        return (inp - mean) / (std + eps) # 只对每张特征图结构内进行标准化，不对全局做，不打破batch之间的分布的独立性

    def forward(self, x):
        '''
        :param x: ppi_estimated, [b,1,h,w]
        :return: edge: edge information described by phase, [b,msfa_size**2,h,w]
        '''
        fft_result = torch.fft.fft2(x)
        phase = torch.angle(fft_result) # TODO: 注意，直接训练angle可能导致梯度爆炸
        x_fake = torch.exp(1j * phase) # 幅值谱部分是1
        phase_map = torch.fft.ifft2(x_fake).real # phase_map: [b,1,h,w]
        phase_map = self.standardize(phase_map) # 保险起见做个标准化试试
        edge = self.PEE(phase_map) # edge: [b,1,h,w] -> [b,c,h,w]
        return edge

class PPIGenerateNew(nn.Module):
    def __init__(self, num_features):
        super(PPIGenerateNew, self).__init__()
        # 深度学习分支
        # 串行写法
        # self.ppig = nn.Sequential(
        #     nn.Conv2d(1, num_features, kernel_size=7, padding=3),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(num_features, num_features, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
        # )

        # 并行写法
        self.ppig_0 = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        )
        self.ppig_1 = nn.Sequential(
             nn.Conv2d(1, num_features, kernel_size=5, padding=2),
             nn.ReLU(inplace=True),
        )
        self.ppig_2 = nn.Sequential(
             nn.Conv2d(1, num_features, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
        )
        self.ppig_fuse = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)

        # 传统PPI filter分支
        self.ppig_traditional = nn.Conv2d(num_features, 1, kernel_size=5, stride=1, padding=2, bias=False, groups=1)
        ppig_traditional_weight = self.get_PPI_filter(5)
        self.ppig_traditional.weight = nn.Parameter(ppig_traditional_weight, requires_grad=False) # fixed params

    @staticmethod
    def get_PPI_filter(size):
        """make a 2D weight kernel suitable for PPI estimation"""
        ligne = [1 / 2, 1, 1, 1, 1 / 2]
        colonne = [1 / 2, 1, 1, 1, 1 / 2]
        PPIFilter = np.zeros(size * size)
        for i in range(size):
            for j in range(size):
                PPIFilter[(j + i * size)] = (ligne[i] * colonne[j] / 16)
        filter_PPI = np.reshape(PPIFilter, (size, size))
        filter_PPI = torch.from_numpy(filter_PPI).float()
        filter_PPI = filter_PPI.view(1, 1, 5, 5)
        return filter_PPI

    def forward(self, x):
        '''
        :param x: mosaic raw, [b,1,h,w]
        :return: ppig_estimated: [b,1,h,w]
        '''
        residual = self.ppig_traditional(x)
        p0 = self.ppig_0(x) # p0: [b,c,h,w]
        p1 = self.ppig_1(x) # p1: [b,c,h,w]
        p2 = self.ppig_2(x) # p2: [b,c,h,w]
        fused = self.ppig_fuse(p0 + p1 + p2) # fused: [b,c,h,w]->[b,1,h,w]
        ppi_estimated = residual + fused # ppi: [b,1,h,w]
        return ppi_estimated

class MPEFormer(nn.Module):
    def __init__(self, stage=2, msfa_size=4):
        super(MPEFormer, self).__init__()
        self.msfa_size = msfa_size

        self.WB_Conv = nn.Conv2d(in_channels=msfa_size ** 2, out_channels=msfa_size ** 2, kernel_size=2 * msfa_size - 1,
                                 stride=1, padding=msfa_size - 1, bias=False, groups=msfa_size ** 2)
        WB_weight = self.get_WB_filter_msfa()
        c1, c2, h, w = self.WB_Conv.weight.data.size() # [out_channel, in_channel, kernel_size_h, kernel_size_w]
        WB_weight = WB_weight.view(1,1,h,w).repeat(c1,c2,1,1)
        self.WB_Conv.weight = nn.Parameter(WB_weight, requires_grad=False)

        self.pg = PPIGenerateNew(msfa_size**2)
        self.edge_ext = PhaseEdgeExtraction(msfa_size**2, use_dwconv=True)

        modules_body = [SSFN(msfa_size=msfa_size, num_blocks=[1,1,1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)

        self.fusion_conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=3, padding=1)

    def get_WB_filter_msfa(self):
        """make a 2D weight bilinear kernel suitable for WB_Conv"""
        size = 2 * self.msfa_size - 1
        ligne = []
        colonne = []
        for i in range(size):
            if (i + 1) <= np.floor(math.sqrt(self.msfa_size ** 2)):
                ligne.append(i + 1)
                colonne.append(i + 1)
            else:
                ligne.append(ligne[i - 1] - 1.0)
                colonne.append(colonne[i - 1] - 1.0)
        BilinearFilter = np.zeros(size * size)
        for i in range(size):
            for j in range(size):
                BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / (self.msfa_size ** 2))
        filter0 = np.reshape(BilinearFilter, (size, size))
        return torch.from_numpy(filter0).float()

    def forward(self, raw, sparse_raw):
        '''
        :param raw: [b,1,h,w]
        :param sparse_raw: [b,c,h,w]
        :return ppi: [b,c,h,w]
        :return h: [b,c,h,w]
        '''
        # raw and sparse_raw should be the same h,w size
        b, c, h_inp, w_inp = sparse_raw.shape
        hb, wb = 32, 32
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb

        # reflect是以左右边界为起点，进行镜像填充（不含起点）
        # F.pad(x,pad,mode): pad内有2n个参数，代表对倒数n个维度进行扩充（4个时候是pad = (左边填充数， 右边填充数， 上边填充数， 下边填充数)）
        # pad以适配多尺度上下采样（确保倍数能除尽）
        x_in_raw = F.pad(raw, [0, pad_w, 0, pad_h], mode='reflect')
        x_in_sparse = F.pad(sparse_raw, [0, pad_w, 0, pad_h], mode='reflect')

        # WB Subbranch
        WB_x = self.WB_Conv(x_in_sparse) # WB_x: [b,c,h,w]

        # Phase-guided Edge Infusion Subbranch
        # Generate PPI from mosaic raw
        ppi = self.pg(x_in_raw) # p, ppi: [b,1,h,w]
        p = ppi.detach()
        # extract edge-related information based on PPI
        high_freq = self.edge_ext(p) # high_freq: [b,c,h,w]

        # Main Subbranch
        h = self.body(x_in_sparse)

        h += WB_x
        h += high_freq
        h = self.fusion_conv(h)

        return ppi[:, :, :h_inp, :w_inp], h[:, :, :h_inp, :w_inp] # 多的部分不要


if __name__ == '__main__':
    model = MPEFormer(stage=2, msfa_size=4)
    raw = torch.randn(1,1,480,512)
    sparse_raw = torch.randn(1,16,480,512)
    input_data = (raw, sparse_raw)
    My_summary(model, input_data=input_data)















