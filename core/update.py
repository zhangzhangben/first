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


def _finite_difference_gradients(disp):
    grad_x = disp.new_zeros(disp.shape)
    grad_y = disp.new_zeros(disp.shape)
    grad_x[:, :, :, :-1] = disp[:, :, :, 1:] - disp[:, :, :, :-1]
    grad_y[:, :, :-1, :] = disp[:, :, 1:, :] - disp[:, :, :-1, :]
    return torch.cat([grad_x, grad_y], dim=1)


def _estimate_uncertainty_from_corr(corr):
    topk = torch.topk(corr, k=min(2, corr.shape[1]), dim=1).values
    if topk.shape[1] == 1:
        peak_gap = torch.zeros_like(topk[:, :1])
    else:
        peak_gap = topk[:, :1] - topk[:, 1:2]
    return torch.sigmoid(-peak_gap)


def _init_relu_conv(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _zero_conv(module):
    if isinstance(module, nn.Conv2d):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _stable_logit(prob, eps=1e-4):
    prob = prob.clamp(eps, 1.0 - eps)
    return torch.log(prob) - torch.log1p(-prob)


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
        self.conv = nn.Conv2d(64 + 256, args.hidden_dims[0] - 1, kernel_size=1, padding=0)
        self.structure_proj = None
        self.conv_with_structure = None

    def _ensure_structure_layers(self, structure_channels, device, dtype):
        if getattr(self, 'structure_proj', None) is not None and getattr(self, 'conv_with_structure', None) is not None:
            return
        proj_channels = int(self.convd2.out_channels)
        self.structure_proj = nn.Sequential(
            nn.Conv2d(structure_channels, proj_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channels, proj_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ).to(device=device, dtype=dtype)
        self.conv_with_structure = nn.Conv2d(
            int(self.conv.in_channels) + proj_channels,
            int(self.conv.out_channels),
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            ).to(device=device, dtype=dtype)
        self.structure_proj.apply(_init_relu_conv)
        with torch.no_grad():
            self.conv_with_structure.weight.zero_()
            self.conv_with_structure.weight[:, :int(self.conv.in_channels)].copy_(self.conv.weight)
            if self.conv_with_structure.bias is not None:
                if self.conv.bias is not None:
                    self.conv_with_structure.bias.copy_(self.conv.bias)
                else:
                    self.conv_with_structure.bias.zero_()

    def forward(self, disp, corr, structure_feat=None):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        if structure_feat is not None:
            self._ensure_structure_layers(structure_feat.shape[1], structure_feat.device, structure_feat.dtype)
            structure_feat = self.structure_proj(structure_feat)
            cor_disp = torch.cat([cor, disp_, structure_feat], dim=1)
            out = F.relu(self.conv_with_structure(cor_disp))
        else:
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
        uncertainty = _estimate_uncertainty_from_corr(corr)
        corr_feat = self.corr_reduce(corr)
        disp_feat = self.disp_reduce(disp)
        gate = self.out(torch.cat([corr_feat, disp_feat, uncertainty], dim=1))
        gate = torch.sigmoid(gate)
        return gate, uncertainty


class LocalStructurePropagationLite(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=128, structure_dim=64, alpha_init=-5.0, max_residual=1.0):
        super().__init__()
        self.center_relation_bias = 6.0
        self.propagation_alpha_logit = nn.Parameter(torch.tensor(float(alpha_init)))
        self.max_residual = float(max_residual)
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim + context_dim + 2, structure_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(structure_dim, structure_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.relation_head = nn.Sequential(
            nn.Conv2d(structure_dim, structure_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(structure_dim, 9, kernel_size=3, padding=1),
        )
        self.grad_head = nn.Sequential(
            nn.Conv2d(structure_dim, structure_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(structure_dim, 2, kernel_size=3, padding=1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(structure_dim, structure_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(structure_dim, 9, kernel_size=3, padding=1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(structure_dim, max(16, structure_dim // 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(16, structure_dim // 2), 1, kernel_size=3, padding=1),
        )
        neighbor_offsets = []
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                neighbor_offsets.append((-float(ox), -float(oy)))
        self.register_buffer('neighbor_offsets', torch.tensor(neighbor_offsets, dtype=torch.float32).view(1, 9, 2, 1, 1))
        self.reset_identity_like()

    def reset_identity_like(self):
        self.structure_encoder.apply(_init_relu_conv)
        self.grad_head[:-1].apply(_init_relu_conv)
        self.offset_head[:-1].apply(_init_relu_conv)
        self.uncertainty_head[:-1].apply(_init_relu_conv)
        self.relation_head[:-1].apply(_init_relu_conv)

        _zero_conv(self.grad_head[-1])
        _zero_conv(self.offset_head[-1])
        _zero_conv(self.uncertainty_head[-1])
        _zero_conv(self.relation_head[-1])
        with torch.no_grad():
            if self.relation_head[-1].bias is not None:
                self.relation_head[-1].bias[4] = self.center_relation_bias

    def initialize_state(self, disp, corr):
        return {
            'uncertainty': _estimate_uncertainty_from_corr(corr),
            'grad': _finite_difference_gradients(disp),
            'offset': disp.new_zeros(disp.shape[0], 9, disp.shape[2], disp.shape[3]),
        }

    def _masked_softmax(self, logits, mask):
        logits = logits.float().masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.float()
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def propagate(self, hidden, context, disp_u, prev_state, margin, grad_scale, offset_scale, blend_scale):
        out_dtype = disp_u.dtype
        margin = max(float(margin), 1e-3)
        grad_scale = float(grad_scale)
        offset_scale = float(offset_scale)
        blend_scale = float(blend_scale)

        with torch.amp.autocast('cuda', enabled=False):
            hidden = hidden.float()
            context = context.float()
            disp_u = disp_u.float()
            prev_uncertainty = prev_state['uncertainty'].float()
            prev_grad = prev_state['grad'].float()
            prev_offset = prev_state['offset'].float()

            fused = self.structure_encoder(torch.cat([hidden, context, disp_u, prev_uncertainty], dim=1))

            grad = prev_grad + torch.tanh(self.grad_head(fused)) * grad_scale
            offset = prev_offset + torch.tanh(self.offset_head(fused)) * offset_scale

            raw_uncertainty = self.uncertainty_head(fused)
            uncertainty = torch.sigmoid(raw_uncertainty + _stable_logit(prev_uncertainty))

            relation_logits = self.relation_head(fused)
            relation_logits = relation_logits - relation_logits.amax(dim=1, keepdim=True)

            b, _, h, w = disp_u.shape
            disp_neighbors = F.unfold(disp_u, kernel_size=3, padding=1).view(b, 9, h, w)
            uncertainty_neighbors = F.unfold(uncertainty, kernel_size=3, padding=1).view(b, 9, h, w)
            valid_neighbors = F.unfold(torch.ones_like(disp_u), kernel_size=3, padding=1).view(b, 9, h, w) > 0

            low_uncertainty = uncertainty_neighbors <= (uncertainty + margin)
            reliable_neighbors = valid_neighbors & low_uncertainty
            center_only = valid_neighbors[:, 4:5].expand_as(valid_neighbors)
            reliable_neighbors = torch.where(reliable_neighbors.any(dim=1, keepdim=True), reliable_neighbors, center_only)

            grad_x = grad[:, 0:1]
            grad_y = grad[:, 1:2]
            dx = self.neighbor_offsets[:, :, 0]
            dy = self.neighbor_offsets[:, :, 1]
            disparity_offset = grad_x * dx + grad_y * dy + offset

            relative_reliability = ((uncertainty - uncertainty_neighbors + margin) / (2.0 * margin)).clamp(0.0, 1.0)
            attention_logits = relation_logits + 2.0 * relative_reliability
            weights = self._masked_softmax(attention_logits, reliable_neighbors)

            propagated_disp = (weights * (disp_neighbors + disparity_offset)).sum(dim=1, keepdim=True)
            propagated_uncertainty = (weights * uncertainty_neighbors).sum(dim=1, keepdim=True)
            propagated_reliability = (weights * relative_reliability).sum(dim=1, keepdim=True)

            propagation_alpha = torch.sigmoid(self.propagation_alpha_logit.float())
            blend = (blend_scale * propagation_alpha * uncertainty * propagated_reliability).clamp(0.0, 1.0)
            residual = (propagated_disp - disp_u).clamp(-self.max_residual, self.max_residual)
            refined_disp = disp_u + blend * residual
            refined_uncertainty = uncertainty + blend * (propagated_uncertainty - uncertainty)
            residual_reg = (blend * residual.abs()).mean()

        return refined_disp.to(out_dtype), {
            'uncertainty': refined_uncertainty.to(out_dtype),
            'grad': grad.to(out_dtype),
            'offset': offset.to(out_dtype),
        }, residual_reg.to(out_dtype)


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

        self.use_loslite_refinement = bool(_cfg_get(args, 'use_loslite_refinement', False))
        self.loslite_hidden_dim = int(_cfg_get(args, 'loslite_hidden_dim', 64))
        self.loslite_uncertainty_margin = float(_cfg_get(args, 'loslite_uncertainty_margin', 0.1))
        self.loslite_propagation_blend = float(_cfg_get(args, 'loslite_propagation_blend', 1.0))
        self.loslite_grad_scale = float(_cfg_get(args, 'loslite_grad_scale', 0.25))
        self.loslite_offset_scale = float(_cfg_get(args, 'loslite_offset_scale', 0.25))
        self.loslite_alpha_init = float(_cfg_get(args, 'loslite_alpha_init', -5.0))
        self.loslite_max_residual = float(_cfg_get(args, 'loslite_max_residual', 1.0))
        self.loslite_module = None
        self.last_loslite_regularizer = None

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

    def _ensure_loslite_module(self, corr, inp):
        if not bool(getattr(self, 'use_loslite_refinement', False)):
            return None
        module = getattr(self, 'loslite_module', None)
        if module is not None:
            return module

        module = LocalStructurePropagationLite(
            hidden_dim=int(self.disp_head.conv[0].in_channels),
            context_dim=int(inp[0].shape[1]),
            structure_dim=int(getattr(self, 'loslite_hidden_dim', 64)),
            alpha_init=float(getattr(self, 'loslite_alpha_init', -5.0)),
            max_residual=float(getattr(self, 'loslite_max_residual', 1.0)),
        )
        module = module.to(device=corr.device, dtype=corr.dtype)
        self.loslite_module = module
        return module

    def forward(self, net, inp, corr, disp, att, structure_state=None):
        loslite_module = self._ensure_loslite_module(corr, inp)
        self.last_loslite_regularizer = disp.new_tensor(0.0)
        motion_features = self.encoder(disp, corr, structure_feat=None)
        motion_features = torch.cat([inp[0], motion_features], dim=1)
        net[0] = self.gru04(att[0], net[0], motion_features)

        base_delta_disp = self.disp_head(net[0])
        delta_disp = base_delta_disp
        gate_module = self._ensure_uncertainty_gate(corr)
        if gate_module is not None:
            gate, uncertainty = gate_module(corr, disp)
            gate_scale = float(getattr(self, 'uncertainty_gate_scale', 1.0))
            gate_bias = float(getattr(self, 'uncertainty_gate_bias', 0.0))
            gate = 1.0 + gate_scale * (gate + gate_bias) * uncertainty
            delta_disp = delta_disp * gate

        next_structure_state = structure_state
        if loslite_module is not None:
            disp_u = disp + delta_disp
            if structure_state is None:
                structure_state = loslite_module.initialize_state(disp_u, corr)
            disp_refined, next_structure_state, residual_reg = loslite_module.propagate(
                hidden=net[0],
                context=inp[0],
                disp_u=disp_u,
                prev_state=structure_state,
                margin=float(getattr(self, 'loslite_uncertainty_margin', 0.1)),
                grad_scale=float(getattr(self, 'loslite_grad_scale', 0.25)),
                offset_scale=float(getattr(self, 'loslite_offset_scale', 0.25)),
                blend_scale=float(getattr(self, 'loslite_propagation_blend', 1.0)),
            )
            delta_disp = disp_refined - disp
            self.last_loslite_regularizer = residual_reg

        mask = .25 * self.mask(net[0])
        return net, mask, delta_disp, next_structure_state
