import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from einops import rearrange
import numbers
# from kt_NEXT import CRNN_MRI
from torch.autograd import Variable
from transforms import fft2c_mri,ifft2c_mri

class DataConsistencyInKspace(nn.Module):


    def __init__(self):
        super(DataConsistencyInKspace, self).__init__()

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def data_consistency(self,k, k0, mask):
        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        out = (1 - mask) * k + k0
        return out

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        k = fft2c_mri(x)
        kdc = self.data_consistency(k,k0,mask)
        x = ifft2c_mri(kdc)


        return x


# --------------------------------------------------------------------------- #
#                             Space dimension reconstruction
# ----------------------------------------------------------------------------#

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):

        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=nn.Identity,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class MetaNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        self.grad_checkpointing = False
        # if ds_stride > 1:
        #     self.downsample = nn.Sequential(
        #         norm_layer(in_chs),
        #         nn.Conv2d(in_chs, out_chs, 3, 1,1),
        #     )
        # else:
        self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        x = self.blocks(x)
        return x


class Space_Attention(nn.Module):
    r""" MetaNeXt
        A PyTorch impl of : `InceptionNeXt: When Inception Meets ConvNeXt`  - https://arxiv.org/pdf/2203.xxxxx.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 9, 3)
        dims (tuple(int)): Feature dimension at each stage. Default: (96, 192, 384, 768)
        token_mixers: Token mixer function. Default: nn.Identity
        norm_layer: Normalziation layer. Default: nn.BatchNorm2d
        act_layer: Activation function for MLP. Default: nn.GELU
        mlp_ratios (int or tuple(int)): MLP ratios. Default: (4, 4, 4, 3)
        head_fn: classifier head
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            in_chans=32,

            depths=(1, 1),
            dims=(32, 32),
            token_mixers=InceptionDWConv2d,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 3),
            drop_rate=0.,
            drop_path_rate=0.,
            ls_init_value=1e-6,
            **kwargs,
    ):
        super().__init__()

        num_stage = len(depths)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage

        self.drop_rate = drop_rate
        # self.stem = nn.Sequential(
        #     nn.Conv2d(in_chans, dims[0],3,1,1),
        #     norm_layer(dims[0])
        # )

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        # feature resolution stages, each consisting of multiple residual blocks
        for i in range(num_stage):
            out_chs = dims[i]
            stages.append(MetaNeXtStage(
                prev_chs,
                out_chs,
                ds_stride=2 if i > 0 else 1,
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                token_mixer=token_mixers[i],
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
            ))
            prev_chs = out_chs
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs
        # self.end = nn.Conv2d(dims[0],in_chans,3,1,1)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward(self, x):
        # x = self.stem(x)
        x = self.stages(x)
        # x = self.end(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# @register_model
# def inceptionnext_tiny(**kwargs):
#     model = Space_Attention(depths=(3,3), dims=(96,96),token_mixers=InceptionDWConv2d,**kwargs)
#     return model

# --------------------------------------------------------------------------- #
#                             Time dimension reconstruction
# ----------------------------------------------------------------------------#

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

ACT_FN = {'gelu': nn.GELU(),'relu' : nn.ReLU(),'lrelu' : nn.LeakyReLU()}

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SpectralBranch(nn.Module):
    def __init__(self,

                 dim,
                 num_heads,
                 bias=False,
                 LayerNorm_type="WithBias"
                 ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, spatial_interaction=None):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        if spatial_interaction is not None:
            q = q * spatial_interaction
            k = k * spatial_interaction
            v = v * spatial_interaction

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)  # (b, c, h, w) -> (b, head, c_, h * w)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h,
                        w=w)  # (b, head, c_, h*w) -> (b, c, h, w)

        out = self.project_out(out)  # (b, c, h, w)
        return out

class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor = 2.66,
                 bias = False,
                 LayerNorm_type = "WithBias",
                 act_fn_name = "gelu"
    ):
        super(Gated_Dconv_FeedForward, self).__init__()
        self.norm = LayerNorm(dim, LayerNorm_type = LayerNorm_type)

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.act_fn = ACT_FN[act_fn_name]

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x

def FFN_FN(ffn_name,dim,ffn_expansion_factor=2.66,bias=False,LayerNorm_type="WithBias",act_fn_name = "gelu"):
    if ffn_name == "Gated_Dconv_FeedForward":
        return Gated_Dconv_FeedForward(
                dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                act_fn_name = act_fn_name
            )

class MixS2_Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ):
        super().__init__()


        self.spectral_branch = SpectralBranch(
            dim,
            num_heads=num_heads,
            bias=False,
            LayerNorm_type="BiasFree"
        )


        self.ffn = Residual(
            FFN_FN(
                dim=dim,
                ffn_name='Gated_Dconv_FeedForward',
                ffn_expansion_factor=2.66,
                bias=False,
                LayerNorm_type="BiasFree"
            )
        )

    def forward(self, x):
        spectral_fea = self.spectral_branch(x)
        spectral_fea = x + spectral_fea
        out = self.ffn(spectral_fea)
        return out

class Time_Attention(nn.Module):
    def __init__(self,
                 num=2,
                 dim=32
                 ):
        super().__init__()

        blocks = []
        for i in range(num):
            blocks.append(MixS2_Block(dim=dim,num_heads=1))
        self.stages = nn.Sequential(*blocks)

    def forward(self, x):
        for stage in self.stages:
            out = stage(x)
        # out = self.stages(x,factor)
        return out

# --------------------------------------------------------------------------- #
#                             Space-Time Transformer reconstruction
# ----------------------------------------------------------------------------#

class DownSample(nn.Module):
    def __init__(self, in_channels,scale_factor, bias=False):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels*(int(1/scale_factor)), 3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,scale_factor, bias=False):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            # blinear interpolate may make results not deterministic
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels//scale_factor, 3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class STTstage(nn.Module):
    def __init__(self,dim=32,dims=(128,128)):
        super().__init__()
        self.spaceattn = Space_Attention(dims=dims)
        self.timeattn = Time_Attention(dim=dim)
        self.end = nn.Conv2d(dim*2,dim,1,bias=False)
        # self.dc = DataConsistencyInKspace()

    def forward(self, x):

        spa = self.spaceattn(x)
        tim = self.timeattn(x)
        out = self.end(torch.cat([spa,tim],dim=1))
        # outdc = self.dc(out,k0,mask)
        return out



# --------------------------------------------------------------------------- #
#                             CRNN  reconstruction
# ----------------------------------------------------------------------------#

class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations

    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    iteration: True or False, to use iteration recurrence or not; if iteration=False: hidden_iteration=None

    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size, dilation, iteration=False):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.iteration = iteration
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=dilation, dilation=dilation)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=dilation, dilation=dilation)
        # add iteration hidden connection
        if self.iteration:
            self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, input, hidden, hidden_iteration=None):
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        if hidden_iteration is not None:
            ih_to_ih = self.ih2ih(hidden_iteration)
            hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        else:
            hidden = self.relu(in_to_hid + hid_to_hid)

        return hidden

class CRNN_MRI(nn.Module):
    """
    CRNN-MRI block in image domain
    RNN evolves over temporal dimension only
    """
    def __init__(self, n_ch, nf=64, ks=3, dilation=2):
        super(CRNN_MRI, self).__init__()
        self.nf = nf
        self.ks = ks

        self.bcrnn_1 = BCRNNlayer(n_ch, nf, ks, dilation=1)
        self.bcrnn_2 = BCRNNlayer(nf, nf, ks, dilation)
        self.bcrnn_3 = BCRNNlayer(nf, nf, ks, dilation)
        self.bcrnn_4 = BCRNNlayer(nf, nf, ks, dilation)

        self.conv4_x = nn.Conv2d(nf, 2, ks, padding=ks//2)

    def forward(self, x, test=False):

        n_batch, n_ch, width, length, n_seq = x.size()

        x = x.permute(4, 0, 1, 2, 3)

        out = self.bcrnn_1(x, None, test)
        out = self.bcrnn_2(out, None, test)
        out = self.bcrnn_3(out, None, test)
        out = self.bcrnn_4(out, None, test)
        out = out.view(-1, self.nf, width, length)
        out = self.conv4_x(out)

        out = out.view(-1, n_batch, 2, width, length)
        out = out.permute(1, 2, 3, 4, 0)

        return out

class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode
               iteration: True if use iteration recurrence and input_iteration is not None; False if input_iteration=None

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size, dilation, iteration=False):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.iteration = iteration
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size, dilation, iteration=self.iteration)

    def forward(self, input, input_iteration=None, test=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h), requires_grad=False).to(torch.device('cuda:1'))
        else:
            hid_init = Variable(torch.zeros(size_h), requires_grad=False).to(torch.device('cuda:1'))

        output_f = []
        output_b = []
        if input_iteration is not None:
            # forward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[i], hidden, input_iteration[i])
                output_f.append(hidden)
            # backward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[nt - i - 1], hidden, input_iteration[nt - i -1])
                output_b.append(hidden)
        else:
            # forward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[i], hidden)
                output_f.append(hidden)
            # backward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[nt - i - 1], hidden)
                output_b.append(hidden)

        output_f = torch.cat(output_f)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b


        output = output.view(nt, nb, self.hidden_size, nx, ny)

        return output

# --------------------------------------------------------------------------- #
#                  CRNN Refined  Space-Time Transformer reconstruction
# ----------------------------------------------------------------------------#
'''
class CST(nn.Module):
    def __init__(self,
                 num_stages=[3,3,3],
                 inchans = 16,
                 nc=5
                 ):
        super().__init__()
        self.dc = DataConsistencyInKspace()
        self.embeding = nn.Conv3d(2,2,3,1,1,bias=False)
        self.down1 = DownSample(in_channels=16)
        self.down2 = DownSample(in_channels=32)
        self.up1 = UpSample(in_channels=64)
        self.up2 = UpSample(in_channels=32)

        self.down3 = DownSample(in_channels=16)
        self.up3 = UpSample(in_channels=32)

        self.down4 = nn.Conv2d(inchans,inchans,3,1,1,bias=False)
        self.up4 = nn.Conv2d(inchans, inchans, 3, 1, 1, bias=False)

        stages1 = []
        for i in range(num_stages[0]):
            stages1.append(STTstage(dim=64,dims=(64,64)))
        self.stages1 = nn.Sequential(*stages1)

        stages2 = []
        for i in range(num_stages[1]):
            stages2.append(STTstage(dim=32,dims=(32,32)))
        self.stages2 = nn.Sequential(*stages2)

        stages3 = []
        for i in range(num_stages[2]):
            stages3.append(STTstage(dim=16,dims=(16,16)))
        self.stages3 = nn.Sequential(*stages3)

        self.end = nn.Conv3d(6,2,3,1,1,bias=False)


        xt_conv_blocks = []
        for i in range(nc):
            xt_conv_blocks.append(CRNN_MRI(6, 64, dilation=3))
        self.xt_conv_blocks = nn.ModuleList(xt_conv_blocks)


    def forward(self, x,k0=None,mask=None):


        x1 = self.embeding(x.permute(0,4,2,3,1)).permute(0,4,2,3,1).contiguous().view(-1,16,128,128)

        x1n = self.down2(self.down1(x1))
        for stage1 in self.stages1:
            x1n = stage1(x1n)
        x1_out = self.up2(self.up1(x1n))+x1
        x1_outdc = self.dc(x1_out,k0,mask)

        x2 = self.down3(x1_outdc.contiguous().view(-1,16,128,128))
        for stage2 in self.stages2:
            x2 = stage2(x2)
        x2_out = self.up3(x2)+x1_outdc.contiguous().view(-1,16,128,128)
        x2_outdc = self.dc(x2_out, k0, mask)


        x3 = self.down4(x2_outdc.contiguous().view(-1,16,128,128))
        for stage3 in self.stages3:
            x3 = stage3(x3)
        x3_out = self.up4(x3)+x2_outdc.contiguous().view(-1,16,128,128)
        x3_outdc = self.dc(x3_out, k0, mask)


        xx = torch.cat([x1_outdc,x2_outdc,x3_outdc],dim=4).permute(0,4,2,3,1)
        for i in range(5):
            # image domain reconstruction
            out = self.xt_conv_blocks[i](xx)
            xx = xx + out

        out = self.end(xx).permute(0,4,2,3,1)
        return out



class CST(nn.Module):
    def __init__(self,
                 num_stages=[2,2,2],
                 inchans = 32,
                 nc=5
                 ):
        super().__init__()
        self.dc = DataConsistencyInKspace()
        self.embeding = nn.Conv2d(inchans,32,3,1,1,bias=False)
        self.down1 = DownSample(in_channels=32)
        self.down2 = DownSample(in_channels=64)
        self.up1 = UpSample(in_channels=128)
        self.up2 = UpSample(in_channels=64)

        self.down3 = DownSample(in_channels=32)
        self.up3 = UpSample(in_channels=64)

        self.down4 = nn.Conv2d(inchans,inchans,3,1,1,bias=False)
        self.up4 = nn.Conv2d(inchans, inchans, 3, 1, 1, bias=False)

        stages1 = []
        for i in range(num_stages[0]):
            stages1.append(STTstage(dim=128,dims=(128,128)))
        self.stages1 = nn.Sequential(*stages1)

        stages2 = []
        for i in range(num_stages[1]):
            stages2.append(STTstage(dim=64,dims=(64,64)))
        self.stages2 = nn.Sequential(*stages2)

        stages3 = []
        for i in range(num_stages[2]):
            stages3.append(STTstage(dim=32,dims=(32,32)))
        self.stages3 = nn.Sequential(*stages3)

        self.end = nn.Conv3d(6,2,3,1,1,bias=False)


        xt_conv_blocks = []
        for i in range(nc):
            xt_conv_blocks.append(CRNN_MRI(2, 64, dilation=3))
        self.xt_conv_blocks = nn.ModuleList(xt_conv_blocks)


    def forward(self, x,k0=None,mask=None):
        x = x.view(-1,32,128,128)

        x1 = self.embeding(x)

        x1n = self.down2(self.down1(x1))
        for stage1 in self.stages1:
            x1n = stage1(x1n)
        x1_out = self.up2(self.up1(x1n))+x1
        x1_outdc = self.dc(x1_out,k0,mask)

        x2 = self.down3(x1_outdc.contiguous().view(-1,32,128,128))
        for stage2 in self.stages2:
            x2 = stage2(x2)
        x2_out = self.up3(x2)+x1_outdc.contiguous().view(-1,32,128,128)
        x2_outdc = self.dc(x2_out, k0, mask)


        x3 = self.down4(x2_outdc.contiguous().view(-1,32,128,128))
        for stage3 in self.stages3:
            x3 = stage3(x3)
        x3_out = self.up4(x3)+x2_outdc.contiguous().view(-1,32,128,128)
        x3_outdc = self.dc(x3_out, k0, mask)


        xx = torch.cat([x1_outdc,x2_outdc,x3_outdc],dim=1).permute(0,4,2,3,1)
        for i in range(5):
            # image domain reconstruction
            out = self.xt_conv_blocks[i](xx)
            xx = xx + out

        out = xx.permute(0,4,2,3,1)
        return out
'''

class CST(nn.Module):
    def __init__(self,
                 num_stages=[2,2,2],
                 inchans = 32
                 ):
        super().__init__()
        self.dc = DataConsistencyInKspace()
        self.embeding = nn.Conv2d(inchans,32,3,1,1,bias=False)

        self.stages1 = nn.ModuleList([])
        for i in range(num_stages[0]):
            self.stages1.append(
                # DownSample(in_channels=32,scale_factor=0.25),
                # DownSample(in_channels=64),
                STTstage(dim=32, dims=(32, 32)),
                # UpSample(in_channels=128,scale_factor=4),
                # UpSample(in_channels=64)
            )


        self.stages2 = nn.ModuleList([])
        for i in range(num_stages[1]):
            self.stages2.append(
                # DownSample(in_channels=32,scale_factor=0.5),
                STTstage(dim=32, dims=(32, 32)),
                # UpSample(in_channels=64,scale_factor=2)
            )


        self.stages3 = nn.ModuleList([])
        for i in range(num_stages[2]):
            self.stages3.append(
                # nn.Conv2d(inchans,inchans,3,1,1,bias=False),
                STTstage(dim=32, dims=(32, 32)),
                # nn.Conv2d(inchans, inchans, 3, 1, 1, bias=False)
            )


        self.end = nn.Conv3d(6,2,3,1,1,bias=False)


        xt_conv_blocks1 = []
        for i in range(3):
            xt_conv_blocks1.append(CRNN_MRI(2, 64, dilation=3))
        self.xt_conv_blocks1 = nn.ModuleList(xt_conv_blocks1)
        xt_conv_blocks2 = []
        for i in range(3):
            xt_conv_blocks2.append(CRNN_MRI(2, 64, dilation=3))
        self.xt_conv_blocks2 = nn.ModuleList(xt_conv_blocks2)
        xt_conv_blocks3 = []
        for i in range(2):
            xt_conv_blocks3.append(CRNN_MRI(2, 64, dilation=3))
        self.xt_conv_blocks3 = nn.ModuleList(xt_conv_blocks3)

    def forward(self, x,k0=None,mask=None):
        x0 = x.view(-1,32,128,128)

        x1 = self.embeding(x0)

        for (stage1) in self.stages1:
            # x11 = down1(x1)
            x11 = stage1(x1)+x1
            # x11 = up1(x11)+x1
            x1 = self.dc(x11.view(-1,16,128,128,2),k0,mask).view(-1,32,128,128)
        x1 = x1.view(-1,16,128,128,2).permute(0,4,2,3,1)
        for i in range(3):
            out = self.xt_conv_blocks1[i](x1)
            x1 = x1 + out # 1,2,128,128,16
            x1 = self.dc(x1.permute(0,4,2,3,1), k0, mask).permute(0,4,2,3,1)

        x2 = x1.permute(0,4,2,3,1).view(-1, 32, 128, 128)
        for (stage2) in self.stages2:
            # x22 = down3(x2)
            x22 = stage2(x2)+x2
            # x22 = up3(x22)+x2
            x2 = self.dc(x22.view(-1,16,128,128,2),k0,mask).view(-1,32,128,128)
        x2 = x2.view(-1, 16, 128, 128, 2).permute(0, 4, 2, 3, 1)
        for i in range(3):
            out = self.xt_conv_blocks2[i](x2)
            x2 = x2 + out
            x2 = self.dc(x2.permute(0,4,2,3,1), k0, mask).permute(0,4,2,3,1)

        x3 = x2.permute(0,4,2,3,1).view(-1, 32, 128, 128)
        for (stage3) in self.stages3:
            # x33 = down4(x3)
            x33 = stage3(x3)+x3
            # x33 = up4(x33)+x3
            x3 = self.dc(x33.view(-1,16,128,128,2),k0,mask).view(-1,32,128,128)
        x3 = x3.view(-1, 16, 128, 128, 2).permute(0, 4, 2, 3, 1)
        for i in range(2):
            out = self.xt_conv_blocks3[i](x3)
            x3 = x3 + out
            x3 = self.dc(x3.permute(0,4,2,3,1), k0, mask).permute(0,4,2,3,1)


        out = self.end(torch.cat([x1,x2,x3],dim=1))
        # out = self.end(x3)
        out = self.dc(out.permute(0,4,2,3,1),k0,mask)
        return out


if __name__ == "__main__":
    def print_networks(net):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        # print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')
    model = CST()
    print_networks(model)
    a = torch.zeros([2,16,128,128,2])
    k = torch.zeros([2, 16, 128, 128, 2])
    m = torch.zeros([2, 16, 128, 128, 2])

    b = model(a,k,m)
    print(b.shape)