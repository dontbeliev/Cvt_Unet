import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Conv_Token_Embeding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Token_Embeding, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=7 // 2,
                               stride=4)
    def forward(self, x):
        x = self.conv1(x)
        return x


class Up_last(nn.Module):
    def __init__(self, in_channels, out_channels, in_c2=0, out_c2=0):
        super(Up_last, self).__init__()
        if in_c2 == 0 and out_c2 == 0:
            in_c2 = in_channels
            out_c2 = out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1,
                                     output_padding=1)
        self.conv = nn.Conv2d(in_c2, out_c2, kernel_size=3, padding=1)
        self.ln = LayerNorm(out_c2)
        self.trans = Transformer(dim=out_c2, proj_kernel=3, kv_proj_stride=2,
                                 depth=2, heads=3, mlp_mult=4, dropout=0.)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv(x)
        x = self.ln(x)
        x = self.trans(x)

        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, heads=1):
        super(Down, self).__init__()
        if in_channels==3:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=7 // 2,
                               stride=4)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3//2,
                               stride=2)
        self.ln1 = LayerNorm(out_channels)
        self.tans1 = Transformer(dim=out_channels, proj_kernel=3, kv_proj_stride=2,
                    depth=depth, heads=heads, mlp_mult=4, dropout=0.)
    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.tans1(x)
        return x

class Skip_func(nn.Module):
    def __init__(self, inc, outc):
        super(Skip_func, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv = nn.Conv2d(inc, outc, (3, 3), padding=1)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, in_c2=0, out_c2=0):
        super(Up, self).__init__()
        if in_c2 == 0 and out_c2 == 0:
            in_c2 = in_channels
            out_c2 = out_channels
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1,
                                     output_padding=1)
        # if in_channels == 192:
        #     self.conv = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        # else:
        self.conv = nn.Conv2d(in_c2, out_c2, kernel_size=3, padding=1)

        self.ln = LayerNorm(out_c2)
        self.trans = Transformer(dim=out_c2, proj_kernel=3, kv_proj_stride=2,
                                 depth=2, heads=3, mlp_mult=4, dropout=0.)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv(x)

        x = self.ln(x)
        x = self.trans(x)

        return x



class Cvt_Unet(nn.Module):
    def __init__(self, numclass=2):
        super(Cvt_Unet, self).__init__()
        self.CTE = Down(3, 64, 1, 1)
        self.down_skip1 = Skip_func(64, 64)
        self.down2 = Down(64, 192, 4, 3)
        # self.down3 = Down(128, 256, 2, 3)
        self.down4 = Down(192, 384, 16, 6)
        self.down_skip2 = Skip_func(192, 192)
        self.down5 = Down(384, 768, 2, 3)
        self.up1 = Up(768, 384, 960, 480)
        self.up2 = Up(480, 240, 496, 248)
        # self.up3 = Up(256, 128)
        self.up4 = Up_last(248, 124, 188, 64)
        self.up5 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(64, numclass, kernel_size=1)

    def forward(self, x):
        x1 = self.CTE(x)

        x1_dskip = self.down_skip1(x1)
        x2 = self.down2(x1)
        x2_dskip = self.down_skip2(x2)
        x4 = self.down4(x2)
        x5 = self.down5(x4)
        x6 = self.up1(x5, x4, x2_dskip)
        x7 = self.up2(x6, x2, x1_dskip)
        x9 = self.up4(x7, x1)
        x10 = self.up5(x9)
        x = self.out(x10)
        return x

# model = Cvt_Unet()
# x = torch.randn(2, 3, 256, 256)
# out = model(x)
# print(out.shape)