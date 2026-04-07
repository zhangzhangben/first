import torch,os,sys
import torch.nn as nn
import torch.nn.functional as F
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.submodule import EdgeNextConvEncoder


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    if hasattr(cfg, 'get'):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
          nn.ReLU(),
          EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
          EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
          nn.Conv2d(input_dim, output_dim, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class BasicMotionEncoder(nn.Module):
    def __init__(self, args, ngroup=8):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (ngroup+1)
        self.convc1 = nn.Conv2d(cor_planes, 256, kernel_size=1, padding=0)
        self.convc2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, kernel_size=7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(64+256, args.hidden_dims[0]-1, kernel_size=1, padding=0)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)

class UncertaintyUpdateGate(nn.Module):
    def __init__(self, corr_channels, hidden_dim=64):
        super().__init__()
        disp_hidden_dim = max(8, hidden_dim // 2)
        self.corr_reduce = nn.Sequential(
            nn.Conv2d(corr_channels, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.disp_reduce = nn.Sequential(
            nn.Conv2d(1, disp_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(hidden_dim + disp_hidden_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )

    def forward(self, corr, disp):
        # A flatter local response profile usually means a more ambiguous match.
        topk = torch.topk(corr, k=min(2, corr.shape[1]), dim=1).values
        if topk.shape[1] == 1:
            peak_gap = torch.zeros_like(topk[:, :1])
        else:
            peak_gap = topk[:, :1] - topk[:, 1:2]
        uncertainty = torch.sigmoid(-peak_gap)

        corr_feat = self.corr_reduce(corr)
        disp_feat = self.disp_reduce(disp)
        gate = self.out(torch.cat([corr_feat, disp_feat, uncertainty], dim=1))
        gate = torch.sigmoid(gate)
        return gate, uncertainty


class RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, x, hx):
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h


class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, small_kernel_size=1, large_kernel_size=3, patch_size=None):
        super(SelectiveConvGRU, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim+hidden_dim, input_dim+hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, *x):
        x = torch.cat(x, dim=1)
        x = self.conv0(x)
        hx = torch.cat([x, h], dim=1)
        hx = self.conv1(hx)
        h = self.small_gru(h, x, hx) * att + self.large_gru(h, x, hx) * (1 - att)

        return h


class BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, volume_dim=8):
        super().__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.volume_dim = volume_dim
        self.encoder = BasicMotionEncoder(args, volume_dim)
        self.use_uncertainty_update_gate = bool(_cfg_get(args, 'use_uncertainty_update_gate', False))
        self.uncertainty_gate_scale = float(_cfg_get(args, 'uncertainty_gate_scale', 1.0))
        self.uncertainty_gate_bias = float(_cfg_get(args, 'uncertainty_gate_bias', 0.0))
        self.uncertainty_gate_hidden_dim = int(_cfg_get(args, 'uncertainty_gate_hidden_dim', 64))
        self.uncertainty_gate = None
        if self.use_uncertainty_update_gate:
            corr_planes = args.corr_levels * (2*args.corr_radius + 1) * (volume_dim + 1)
            self.uncertainty_gate = UncertaintyUpdateGate(corr_planes, hidden_dim=self.uncertainty_gate_hidden_dim)

        self.gru04 = SelectiveConvGRU(hidden_dim, hidden_dim*2)
        self.disp_head = DispHead(hidden_dim, 256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            )

    def _ensure_uncertainty_gate(self, corr):
        if not bool(getattr(self, 'use_uncertainty_update_gate', False)):
            return None
        gate = getattr(self, 'uncertainty_gate', None)
        if gate is not None:
            return gate

        gate_hidden_dim = int(getattr(self, 'uncertainty_gate_hidden_dim', 64))
        gate = UncertaintyUpdateGate(corr.shape[1], hidden_dim=gate_hidden_dim)
        gate = gate.to(device=corr.device, dtype=corr.dtype)
        self.uncertainty_gate = gate
        return gate

    def forward(self, net, inp, corr, disp, att):
        motion_features = self.encoder(disp, corr)
        motion_features = torch.cat([inp[0], motion_features], dim=1)
        net[0] = self.gru04(att[0], net[0], motion_features)

        delta_disp = self.disp_head(net[0])
        gate_module = self._ensure_uncertainty_gate(corr)
        if gate_module is not None:
            gate, uncertainty = gate_module(corr, disp)
            gate_scale = float(getattr(self, 'uncertainty_gate_scale', 1.0))
            gate_bias = float(getattr(self, 'uncertainty_gate_bias', 0.0))
            gate = 1.0 + gate_scale * (gate + gate_bias) * uncertainty
            delta_disp = delta_disp * gate

        mask = .25 * self.mask(net[0])
        return net, mask, delta_disp

