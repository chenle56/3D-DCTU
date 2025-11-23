
import math
from torchsummary import summary
import torch
import torch.nn as nn
import numpy as np
# from thop import profile
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath
from deformable_LKA.deformable_LKA3d import *
from dcn.modules.deform_conv import *
import skimage
import scipy.io

def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0],H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], C) #这里的维度跟源码不一样
    return windows

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.shift_size=[size//2 for size in window_size]
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True) # q,k,v是通过一个全连接层实现的，dim->3*dim
        mesh_args = torch.meshgrid.__kwdefaults__  #抄的3d swin——unetr

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))

        #define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), self.n_heads)  # [2*Mh-1 *2*Mw-1，nH]
        )
        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing = "ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))

        coords_flatten = torch.flatten(coords, 1)  # [2, Mh * Mw]
        # [2, Mh * Mw, 1] - [2, 1, Mh * Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.linear = nn.Linear(self.input_dim, self.output_dim) # q,k,v是通过一个全连接层实现的，dim->3*dim
        trunc_normal_(self.relative_position_bias_table, std = 0.02)

    def create_mask(self, D, H, W, window_size, shift_size):
        # calculate attention mask for SW-MSA
        window_size, shift_size = get_window_size((D, H, W), window_size, shift_size)

        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=self.relative_position_bias_table.device)  # [1, Hp, Wp, 1]

        d_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        h_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        w_slices = (slice(0, -window_size[2]),
                    slice(-window_size[2], -shift_size[2]),
                    slice(-shift_size[2], None))
        cnt = 0

        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """

        b, d, h, w, c = x.shape
        self.window_size, self.shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)

        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size[0]//2), -(self.window_size[1]//2), -(self.window_size[2]//2)), dims=(1,2,3))
        x = rearrange(x, 'b (w1 p1) (w2 p2) (w3 p3) c -> b w1 w2 w3 p1 p2 p3 c', p1=self.window_size[0], p2=self.window_size[1], p3=self.window_size[2])
        d_windows = x.size(1)
        h_windows = x.size(2)
        w_windows = x.size(3)
        x = rearrange(x, 'b w1 w2 w3 p1 p2 p3 c -> b (w1 w2 w3) (p1 p2 p3) c', p1=self.window_size[0], p2=self.window_size[1], p3=self.window_size[2])
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        n = self.window_size[0]*self.window_size[1]*self.window_size[2]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # [nH, Mh*Mw, Mh*Mw]
        # Adding learnable relative embedding
        # sim = [1, batch, num_windows, Md * Mh * Mw, Md * Mh * Mw], h=1, p=Md * Mh * Mw, q=Md * Mh * Mw
        sim = sim + rearrange(relative_position_bias, 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask =self.create_mask(d_windows*self.window_size[0],h_windows*self.window_size[1], w_windows*self.window_size[2], self.window_size, self.shift_size)
            sim = sim+rearrange(attn_mask, 'h p q -> 1 1 h p q')  #这里可能有问题
        probs = nn.functional.softmax(sim, dim=-1)
        # attn @ v
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        # window_reverse
        output = rearrange(output, 'b (w1 w2 w3) (p1 p2 p3) c -> b (w1 p1) (w2 p2) (w3 p3) c', w1=d_windows, p1=self.window_size[0], w2=h_windows, p2=self.window_size[1])

        if self.type != 'W': output = torch.roll(output, shifts = (self.window_size[0] // 2, self.window_size[1] // 2, self.window_size[2] // 2),
                                                 dims = (1, 2, 3))
        return output


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path,
                                 self.type, self.input_resolution)
        self.conv1_1 = nn.Conv3d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv3d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)

        self.DCN_block=nn.Sequential(
            DeformConvPack(self.conv_dim, self.conv_dim, 3, stride=1, padding=1),
            nn.ReLU(True),
            DeformConvPack(self.conv_dim, self.conv_dim, 3, stride=1, padding=1)
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.DCN_block(conv_x) + conv_x
        trans_x = Rearrange('b c d h w -> b d h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b d h w c -> b c d h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res

        return x


class SCUNet(nn.Module):

    def __init__(self, in_nc=1, config=[2, 2, 2, 2, 2, 2, 2], dim=64, drop_path_rate=0.1, input_resolution=[64,64,64]):
        super(SCUNet, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = [8, 8, 8]

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv3d(in_nc, dim, 3, 1, 1, bias=False)]
        begin = 0
        self.m_down1 = [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution)
                        for i in range(config[0])] + \
                       [nn.Conv3d(dim, 2 * dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', [size // 2 for size in input_resolution])
                        for i in range(config[1])] + \
                       [nn.Conv3d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', [size // 4 for size in input_resolution])
                        for i in range(config[2])] + \
                       [nn.Conv3d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                      'W' if not i % 2 else 'SW', [size // 8 for size in input_resolution])
                       for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose3d(8 * dim, 4 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', [size // 4 for size in input_resolution])
                      for i in range(config[4])]

        begin += config[4]
        self.m_up2 = [nn.ConvTranspose3d(4 * dim, 2 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', [size // 2 for size in input_resolution])
                      for i in range(config[5])]

        begin += config[5]
        self.m_up1 = [nn.ConvTranspose3d(2 * dim, dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution)
                      for i in range(config[6])]

        self.m_tail = [nn.Conv3d(dim, in_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)
        # self.apply(self._init_weights)

    # def forward(self, x0):
    def forward(self, x0):
        d, h, w = x0.size()[-3:]
        paddingBack = int(np.ceil(d / 32) * 32 - d)
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad3d((0, paddingRight, 0, paddingBottom, 0, paddingBack))(x0)
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        x = x[..., :d, :h, :w]
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(torch.cuda.is_available())
    net = SCUNet()
    net.cuda()
    x = torch.randn((1, 1, 64, 64, 64)).cuda()
    x = net(x)
    print(x.shape)
    print(x[..., :10, :10].shape)