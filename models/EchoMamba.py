import torch
from torch import nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from models.utils import *
from models.config_setting import setting_config
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from torchvision.ops import DeformConv2d

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


config = setting_config
log_dir = os.path.join(config.work_dir, 'log')
global logger
logger = get_logger('train', log_dir)
log_config_info(config, logger)


# ----------GAB---------------------------------------------------------------------------------------------
class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1, 2, 5, 7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=group_size + 1)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=group_size + 1)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                      dilation=d_list[2], groups=group_size + 1)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                      dilation=d_list[3], groups=group_size + 1)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1)
        )

    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x


# ------SS2D----------------------------------------------------------------------------

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        # print("out shape:", out.shape)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSS(nn.Module):
    def __init__(
            self,
            dim,
            drop_path: float = 0,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            h=14,
            w=8,
            **kwargs,
    ):
        super().__init__()

        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.self_attention = SS2D(d_model=dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)


    def forward(self, input: torch.Tensor):
        x = self.pre_norm(input)
        x = x.permute(0, 2, 3, 1)
        x =self.drop_path(self.self_attention(x))
        x =x.permute(0, 3, 1, 2)
        x = self.post_norm(x)
        return x


# ------ResVMamba----------------------------------------------------------------------------


class ResVM_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba=VSS(
            dim=input_dim,
            drop_path=0,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            d_state=16,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        # self.conv=nn.Conv2d(input_dim, output_dim, 3, stride=1, padding=1)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim

        x_mamba=x * self.skip_scale + self.mamba(x)
        x_mamba = x_mamba.permute(0, 2, 3, 1)
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        x_mamba = x_mamba.permute(0, 3, 1, 2)

        return x_mamba


# ------AFE----------------------------------------------------------------------------

class AFE_Layer(nn.Module):
    def __init__(self, in_channels, offset_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(AFE_Layer, self).__init__()

        self.offsets_conv = nn.Conv2d(in_channels, offset_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding)

        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding)
        self.norm_layer = LayerNorm(out_channels, eps=1e-6, data_format='channels_first')
        self.silu = nn.SiLU()
        self.depthwise1= nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1,groups=in_channels )
        # self.depthwise2 = nn.Conv2d(in_channels, in_channels, 5, stride=1, padding=2, groups=in_channels)
        # self.depthwise3 = nn.Conv2d(in_channels, in_channels, 7, stride=1, padding=3, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1 ) # 1x1卷积
    def forward(self, x):

        offsets = self.offsets_conv(x)
        x1 = self.deform_conv(x, offsets)
        x1 = self.norm_layer(x1)
        x2 = self.depthwise2(x)
        x4 =  self.pointwise( x2 + x)
        x4 = self.silu(x4)
        out = x1 + x4
        return out


class EchoMamba(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc', bridge='03', gt_ds=True):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),

        )
        self.encoder2 = nn.Sequential(
            ResVM_Layer(input_dim=c_list[0], output_dim=c_list[1]),
        )
        self.encoder3 = nn.Sequential(
            AFE_Layer(  c_list[1], 18 , c_list[2] ,3, stride=1, padding=1 )
        )
        self.encoder4 = nn.Sequential(
            ResVM_Layer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            AFE_Layer(c_list[3], 18, c_list[4], 3, stride=1, padding=1)
        )
        self.encoder6 = nn.Sequential(
            ResVM_Layer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge == '03':
            print('**********************bridge=***********', bridge)
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
            print('**************group_aggregation_bridge was used***********')
            log_info = f'03_Bridge was used'
            print(log_info)
            logger.info(log_info)

        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')

        self.decoder1 = nn.Sequential(
            ResVM_Layer(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.decoder2 = nn.Sequential(
            AFE_Layer(c_list[4]*2, 18, c_list[3], 3, stride=1, padding=1)
        )
        self.decoder3 = nn.Sequential(
            ResVM_Layer(input_dim=c_list[3]*2, output_dim=c_list[2])
        )
        self.decoder4 = nn.Sequential(
            AFE_Layer(c_list[2]*2, 18, c_list[1], 3, stride=1, padding=1)
        )
        self.decoder5 = nn.Sequential(
            ResVM_Layer(input_dim=c_list[1]*2, output_dim=c_list[0])
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0]*2, num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        if self.gt_ds:
            gt_pre5 = self.gt_conv1(out5)
            t5 = self.GAB5(out, t5, gt_pre5)
            gt_pre5 = F.interpolate(gt_pre5, size=(112, 112), mode='bilinear', align_corners=True)
        else:
            t5 = self.GAB5(out, t5)
        out5 = torch.cat([out5, t5], dim=1)  # b, c4, H/32, W/32


        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16

        pad = nn.ZeroPad2d(padding=(1, 2, 1, 2))  # (left, right, top, bottom)
        pad1 = pad(out4)
        start_index = (9 - 7) // 2  # 1
        end_index = start_index + 7  # 8
        # center  7x7
        out4 = pad1[:, :, start_index:end_index, start_index:end_index]
        if self.gt_ds:
            gt_pre4 = self.gt_conv2(out4)
            t4 = self.GAB4(t5, t4, gt_pre4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)
        else:
            t4 = self.GAB4(t5, t4)
        out4 =torch.cat([out4, t4], dim=1)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        if self.gt_ds:
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.GAB3(t4, t3, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)
        else:
            t3 = self.GAB3(t4, t3)
        out3 = torch.cat([out3, t3], dim=1)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.GAB2(t3, t2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            t2 = self.GAB2(t3, t2)
        out2 = torch.cat([out2, t2], dim=1)

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        if self.gt_ds:
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            t1 = self.GAB1(t2, t1)
        out1 = torch.cat([out1, t1], dim=1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        if self.gt_ds:
            return (torch.sigmoid(gt_pre5), torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2),
                    torch.sigmoid(gt_pre1)), torch.sigmoid(out0)
        else:
            return torch.sigmoid(out0)