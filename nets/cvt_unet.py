import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(x))
        return out


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=1, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out


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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.dw = ConvBNReLU(dim * mult, dim * mult, kernel_size=3, groups=dim * mult)
        self.dw1 = ConvBNReLU(dim * mult, dim * mult, kernel_size=5, groups=dim * mult)
        self.dw2 = ConvBNReLU(dim * mult, dim * mult, kernel_size=7, groups=dim * mult)
        self.net1 = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1)
        )

    def forward(self, x):
        x = self.net(x)
        x1 = self.dw(x)
        x2 = self.dw1(x)
        x3 = self.dw2(x)
        x = self.net1(x1 + x2 + x3)
        # x = self.net1(x)
        return x


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
        self.cbam = CBAM(dim)
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + self.cbam(x)
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



class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1,
                                     output_padding=1)
        if in_channels == 192:
            self.conv = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.ln = LayerNorm(out_channels)
        self.trans = Transformer(dim=out_channels, proj_kernel=3, kv_proj_stride=2,
                                 depth=2, heads=3, mlp_mult=4, dropout=0.)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv(x)
        x = self.ln(x)
        x = self.trans(x)
        return x

class Cvt_Unet_1(nn.Module):   # acc:99.34
    def __init__(self, numclass=1000):
        super(Cvt_Unet_1, self).__init__()
        self.CTE = Down(3, 64, 1, 1)
        self.down2 = Down(64, 128, 2, 3)
        self.down3 = Down(128, 256, 2, 3)
        self.down4 = Down(256, 512, 10, 6)
        self.down5 = Down(512, 1024, 2, 3)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.up5 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(64, numclass, kernel_size=1)
    def forward(self, x):
        x1 = self.CTE(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.up5(x9)
        x = self.out(x10)
        return x


class Cvt_Unet(nn.Module):
    def __init__(self, numclass=2):
        super(Cvt_Unet, self).__init__()
        self.CTE = Down(3, 64, 1, 1)
        self.down2 = Down(64, 192, 2, 3)
        # self.down3 = Down(128, 256, 2, 3)
        self.down4 = Down(192, 384, 10, 6)
        self.down5 = Down(384, 768, 2, 3)
        self.up1 = Up(768, 384)
        self.up2 = Up(384, 192)
        # self.up3 = Up(256, 128)
        self.up4 = Up(192, 64)
        self.up5 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(64, numclass, kernel_size=1)
    def forward(self, x):
        x1 = self.CTE(x)
        x2 = self.down2(x1)
        # x3 = self.down3(x2)
        x4 = self.down4(x2)
        x5 = self.down5(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x2)
        # x8 = self.up3(x7, x2)
        x9 = self.up4(x7, x1)
        x10 = self.up5(x9)
        x = self.out(x10)
        return x


if __name__ == '__main__':
    model = Cvt_Unet()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(out.shape)