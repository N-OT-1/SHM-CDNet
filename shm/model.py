# SHM core extracted from the frozen implementation. See LICENSE.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import PixelData
from mmseg.registry import MODELS
import math
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print('Warning: scipy not available, edge generation will use torch operations')

def sliding_blend_weight(height, width, mode='uniform', sigma=0.5, min_weight=0.1, device=None, dtype=None):
    """Build a positive 2D window for overlap-normalized slide inference."""
    if mode == 'uniform':
        return torch.ones((1, 1, height, width), device=device, dtype=dtype)
    if mode != 'gaussian':
        raise ValueError(f'Unsupported slide blend mode={mode}; expected uniform or gaussian')
    sigma = float(sigma)
    min_weight = float(min_weight)
    if sigma <= 0.0:
        raise ValueError('slide blend sigma must be positive')
    if not 0.0 < min_weight <= 1.0:
        raise ValueError('slide blend min_weight must be in (0, 1]')
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    weight = torch.exp(-0.5 * ((y[:, None] / sigma) ** 2 + (x[None, :] / sigma) ** 2))
    weight = weight / weight.max().clamp(min=torch.finfo(weight.dtype).eps)
    return weight.clamp(min=min_weight).unsqueeze(0).unsqueeze(0)

def mask_to_edge(mask, edge_width=3, ignore_index=255):
    """
    Convert segmentation mask to edge map.

    Args:
        mask: (B, H, W) segmentation mask with values {0, 1, 255}
        edge_width: Width of edge (default: 3)
        ignore_index: Ignore index value (default: 255)

    Returns:
        edge_map: (B, H, W) binary edge map, 1=edge, 0=non-edge
    """
    B, H, W = mask.shape
    device = mask.device
    edge_map = torch.zeros_like(mask, dtype=torch.float32)
    ignore_mask = mask == ignore_index
    valid_mask = ~ignore_mask
    for b in range(B):
        mask_b = mask[b]
        valid_b = valid_mask[b]
        if not valid_b.any():
            continue
        unique_classes = torch.unique(mask_b[valid_b])
        for cls in unique_classes:
            if cls == ignore_index:
                continue
            cls_mask = (mask_b == cls).float().unsqueeze(0).unsqueeze(0)
            if SCIPY_AVAILABLE:
                cls_mask_np = cls_mask.squeeze().cpu().numpy().astype(np.uint8)
                kernel = np.ones((edge_width, edge_width), dtype=np.uint8)
                dilated = ndimage.binary_dilation(cls_mask_np, structure=kernel)
                edge = dilated.astype(np.float32) - cls_mask_np.astype(np.float32) > 0
                edge_tensor = torch.from_numpy(edge.astype(np.float32)).to(device)
            else:
                kernel_size = edge_width * 2 + 1
                padding = edge_width
                dilated = F.max_pool2d(cls_mask, kernel_size=kernel_size, stride=1, padding=padding)
                edge = dilated.squeeze() - cls_mask.squeeze() > 0
                edge_tensor = edge.float()
            edge_map[b] = torch.maximum(edge_map[b], edge_tensor)
    return edge_map

def edge_bce_loss(edge_logits, edge_gt, pos_weight=5.0, ignore_index=255):
    """
    Binary Cross Entropy loss for edge prediction.

    Args:
        edge_logits: (B, 1, H, W) edge prediction logits
        edge_gt: (B, H, W) ground truth edge map (1=edge, 0=non-edge)
        pos_weight: Weight for positive (edge) pixels (default: 5.0)
        ignore_index: Ignore index value (not used for edges, kept for compatibility)

    Returns:
        loss: Scalar edge BCE loss
    """
    edge_logits = edge_logits.squeeze(1)
    edge_gt = edge_gt.float()
    loss = F.binary_cross_entropy_with_logits(edge_logits, edge_gt, pos_weight=torch.tensor(pos_weight, device=edge_logits.device, dtype=edge_logits.dtype), reduction='mean')
    return loss
try:
    from .mamba_vision import MambaVisionMixer, Attention, Mlp, LayerNorm2d, Block, ConvBlock, Downsample, PatchEmbed, MambaVisionLayer
except ImportError:
    from mamba_vision import MambaVisionMixer, Attention, Mlp, LayerNorm2d, Block, ConvBlock, Downsample, PatchEmbed, MambaVisionLayer

def build_norm_layer(num_channels, norm_cfg=None):
    """Build 2D normalization layer from config."""
    norm_cfg = norm_cfg or dict(type='BN')
    norm_type = norm_cfg.get('type', 'BN').upper()
    eps = norm_cfg.get('eps', 1e-05)
    momentum = norm_cfg.get('momentum', 0.1)
    if norm_type == 'BN':
        return nn.BatchNorm2d(num_channels, eps=eps, momentum=momentum)
    if norm_type == 'SYNCBN':
        return nn.SyncBatchNorm(num_channels, eps=eps, momentum=momentum)
    if norm_type == 'GN':
        num_groups = int(norm_cfg.get('num_groups', 32))
        while num_groups > 1 and num_channels % num_groups != 0:
            num_groups //= 2
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)
    raise ValueError(f'Unsupported norm type: {norm_type}')

class MambaVisionBackbone(BaseModule):
    """
    MambaVision Backbone for feature extraction
    Returns multi-scale features: {C1, C2, C3, C4}
    """

    def __init__(self, dim=80, in_dim=32, depths=(1, 3, 8, 4), window_size=(8, 8, 64, 32), mlp_ratio=4.0, num_heads=(2, 4, 8, 16), drop_path_rate=0.2, in_chans=3, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, layer_scale=None, layer_scale_conv=None, out_indices=(0, 1, 2, 3), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if i == 0 or i == 1 else False
            level = MambaVisionLayer(dim=int(dim * 2 ** i), depth=depths[i], num_heads=num_heads[i], window_size=window_size[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, conv=conv, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], downsample=i < 3, layer_scale=layer_scale, layer_scale_conv=layer_scale_conv, transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0 else list(range(depths[i] // 2, depths[i])))
            self.levels.append(level)
        self.dims = [int(dim * 2 ** i) for i in range(len(depths))]

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            features: List of [C1, C2, C3, C4] with shapes:
                C1: (B, 80, H/4, W/4)
                C2: (B, 160, H/8, W/8)
                C3: (B, 320, H/16, W/16)
                C4: (B, 640, H/32, W/32)
        """
        ref_param = next(self.patch_embed.parameters())
        model_dtype = ref_param.dtype
        model_device = ref_param.device
        if x.dtype != model_dtype:
            x = x.to(dtype=model_dtype)
        if x.device != model_device:
            x = x.to(device=model_device)
        x = self.patch_embed(x)
        outs = []
        for i, level in enumerate(self.levels):
            x, o = level(x)
            if i in self.out_indices:
                outs.append(o)
        return outs

class DifferenceModule(BaseModule):
    """
    Difference Fusion Module
    Combines: |D1 - D2| + D1 * D2 for comprehensive change modeling
    """

    def __init__(self, in_channels, out_channels=128, norm_cfg=None, align_mode='pre', use_product_feature=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.align_mode = align_mode
        self.use_product_feature = bool(use_product_feature)
        if align_mode not in ('pre', 'post'):
            raise ValueError(f"Unsupported align_mode={align_mode}, expected 'pre' or 'post'")
        if align_mode == 'pre':
            self.align_conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.align_conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.norm1 = build_norm_layer(out_channels, norm_cfg)
            self.norm2 = build_norm_layer(out_channels, norm_cfg)
            diff_channels = out_channels
        else:
            self.align_conv1 = None
            self.align_conv2 = None
            self.norm1 = None
            self.norm2 = None
            diff_channels = in_channels
        self.fusion_conv = nn.Sequential(nn.Conv2d(diff_channels * 2, out_channels, 3, padding=1, bias=False), build_norm_layer(out_channels, norm_cfg), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), build_norm_layer(out_channels, norm_cfg), nn.ReLU(inplace=True))

    def forward(self, f1, f2):
        """
        Args:
            f1, f2: (B, C, H, W) features from two time phases
        Returns:
            fused: (B, out_channels, H, W) difference features
        """
        if self.align_mode == 'pre':
            d1 = F.relu(self.norm1(self.align_conv1(f1)))
            d2 = F.relu(self.norm2(self.align_conv2(f2)))
        else:
            d1 = f1
            d2 = f2
        abs_diff = torch.abs(d1 - d2)
        mult = d1 * d2
        if not self.use_product_feature:
            mult = torch.zeros_like(mult)
        combined = torch.cat([abs_diff, mult], dim=1)
        fused = self.fusion_conv(combined)
        return fused

class GatedDifferenceModule(BaseModule):
    """
    Gated Difference Module with Change-Aware Gate

    Core idea:
        Instead of: D = |F1 - F2|
        Use: G = sigmoid(Conv([F1, F2, D]))
             D = |F1 - F2| * G
    """

    def __init__(self, in_channels, out_channels=128, gate_type='difference', gate_channels='spatial', norm_cfg=None, align_mode='pre', use_diff_in_gate=True, use_product_feature=True, gate_arrangement='parallel', init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.gate_type = gate_type
        self.gate_channels = gate_channels
        if gate_arrangement not in ('parallel', 'spatial_then_channel', 'channel_then_spatial'):
            raise ValueError(f'Unsupported gate_arrangement={gate_arrangement}')
        if gate_arrangement != 'parallel' and gate_channels != 'channel_spatial':
            raise ValueError('Serial gating requires both spatial and channel gates')
        self.gate_arrangement = gate_arrangement
        self.align_mode = align_mode
        self.use_diff_in_gate = use_diff_in_gate
        self.use_product_feature = bool(use_product_feature)
        if align_mode not in ('pre', 'post'):
            raise ValueError(f"Unsupported align_mode={align_mode}, expected 'pre' or 'post'")
        if gate_channels not in ('spatial', 'channel', 'channel_spatial'):
            raise ValueError(f"Unsupported gate_channels={gate_channels}, expected one of ('spatial', 'channel', 'channel_spatial')")
        if 1 > 1 and gate_channels != 'channel_spatial':
            raise ValueError('gate_rank > 1 requires gate_channels="channel_spatial"')
        if align_mode == 'pre':
            self.align_conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.align_conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.norm1 = build_norm_layer(out_channels, norm_cfg)
            self.norm2 = build_norm_layer(out_channels, norm_cfg)
            diff_channels = out_channels
        else:
            self.align_conv1 = None
            self.align_conv2 = None
            self.norm1 = None
            self.norm2 = None
            diff_channels = in_channels
        gate_in_channels = diff_channels * (3 if use_diff_in_gate else 2)
        self.spatial_gate_conv = None
        if gate_channels in ('spatial', 'channel_spatial'):
            self.spatial_gate_conv = nn.Sequential(nn.Conv2d(gate_in_channels, diff_channels, 3, padding=1, bias=False), build_norm_layer(diff_channels, norm_cfg), nn.ReLU(inplace=True), nn.Conv2d(diff_channels, max(diff_channels // 2, 16), 3, padding=1, bias=False), build_norm_layer(max(diff_channels // 2, 16), norm_cfg), nn.ReLU(inplace=True), nn.Conv2d(max(diff_channels // 2, 16), 1, 1), nn.Sigmoid())
        self.channel_gate_conv = None
        if gate_channels in ('channel', 'channel_spatial'):
            hidden = max(diff_channels // 2, 16)
            self.channel_gate_conv = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(gate_in_channels, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, diff_channels, 1), nn.Sigmoid())
        self.fusion_conv = nn.Sequential(nn.Conv2d(diff_channels * 2, out_channels, 3, padding=1, bias=False), build_norm_layer(out_channels, norm_cfg), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), build_norm_layer(out_channels, norm_cfg), nn.ReLU(inplace=True))

    def forward(self, f1, f2):
        if self.align_mode == 'pre':
            d1 = F.relu(self.norm1(self.align_conv1(f1)))
            d2 = F.relu(self.norm2(self.align_conv2(f2)))
        else:
            d1 = f1
            d2 = f2
        diff = torch.abs(d1 - d2)
        mult = d1 * d2
        if not self.use_product_feature:
            mult = torch.zeros_like(mult)
        if self.use_diff_in_gate:
            gate_input = torch.cat([d1, d2, diff], dim=1)
        else:
            gate_input = torch.cat([d1, d2], dim=1)
        gate = 1.0
        if self.gate_arrangement == 'spatial_then_channel':
            spatial = self.spatial_gate_conv(gate_input)
            channel = self.channel_gate_conv(gate_input * spatial)
            gate = spatial * channel
        elif self.gate_arrangement == 'channel_then_spatial':
            channel = self.channel_gate_conv(gate_input)
            blocks = gate_input.shape[1] // channel.shape[1]
            spatial = self.spatial_gate_conv(gate_input * channel.repeat(1, blocks, 1, 1))
            gate = channel * spatial
        else:
            if self.spatial_gate_conv is not None:
                gate = gate * self.spatial_gate_conv(gate_input)
            if self.channel_gate_conv is not None:
                gate = gate * self.channel_gate_conv(gate_input)
        if self.gate_type == 'feature':
            d1_gated = d1 * gate
            d2_gated = d2 * gate
            abs_diff = torch.abs(d1_gated - d2_gated)
        else:
            abs_diff = diff * gate
        combined = torch.cat([abs_diff, mult], dim=1)
        fused = self.fusion_conv(combined)
        return fused

class FPNDecoder(BaseModule):
    """
    FPN-style decoder for change detection.

    Supports two top-down fusion modes:
    - add: legacy lateral add fusion
    - concat: concat + conv fusion (stronger semantic alignment)

    Optionally adds deep supervision heads on P2/P3/P4.
    """

    def __init__(self, in_channels=[128, 128, 128, 128], fpn_dim=128, num_classes=2, use_edge_head=True, norm_cfg=None, fuse_mode='add', use_deep_supervision=False, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if fuse_mode not in ('add', 'concat'):
            raise ValueError(f"Unsupported fuse_mode={fuse_mode}, expected 'add' or 'concat'")
        self.fuse_mode = fuse_mode
        self.use_deep_supervision = use_deep_supervision
        self.lateral_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels[i], fpn_dim, 1, bias=False), build_norm_layer(fpn_dim, norm_cfg), nn.ReLU(inplace=True)) for i in range(len(in_channels))])
        self.fpn_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False), build_norm_layer(fpn_dim, norm_cfg), nn.ReLU(inplace=True)) for _ in range(len(in_channels) - 1)])
        self.concat_fuse_convs = None
        if fuse_mode == 'concat':
            self.concat_fuse_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(fpn_dim * 2, fpn_dim, 1, bias=False), build_norm_layer(fpn_dim, norm_cfg), nn.ReLU(inplace=True), nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False), build_norm_layer(fpn_dim, norm_cfg), nn.ReLU(inplace=True)) for _ in range(len(in_channels) - 1)])
        self.seg_head = nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False), build_norm_layer(fpn_dim, norm_cfg), nn.ReLU(inplace=True), nn.Conv2d(fpn_dim, 64, 3, padding=1, bias=False), build_norm_layer(64, norm_cfg), nn.ReLU(inplace=True), nn.Conv2d(64, num_classes, 1))
        self.aux_heads = None
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList([nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False), build_norm_layer(fpn_dim, norm_cfg), nn.ReLU(inplace=True), nn.Conv2d(fpn_dim, num_classes, 1)) for _ in range(3)])
        self.use_edge_head = use_edge_head
        if use_edge_head:
            self.edge_head = nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim // 2, 3, padding=1, bias=False), build_norm_layer(fpn_dim // 2, norm_cfg), nn.ReLU(inplace=True), nn.Conv2d(fpn_dim // 2, 32, 3, padding=1, bias=False), build_norm_layer(32, norm_cfg), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))

    def _fuse_topdown(self, lateral, topdown, stage_idx):
        if self.fuse_mode == 'concat':
            fused = torch.cat([lateral, topdown], dim=1)
            return self.concat_fuse_convs[stage_idx](fused)
        return lateral + topdown

    def forward(self, features):
        """
        Args:
            features: List of [F1, F2, F3, F4] after difference module
        Returns:
            seg_logits: (B, num_classes, H/4, W/4)
            edge_logits: (B, 1, H/4, W/4) or None
            aux_logits: list of auxiliary logits [P2, P3, P4] if enabled else []
        """
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        p4 = laterals[3]
        p = F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = self._fuse_topdown(laterals[2], p, stage_idx=2)
        p3 = self.fpn_convs[2](p3)
        p = F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=False)
        p2 = self._fuse_topdown(laterals[1], p, stage_idx=1)
        p2 = self.fpn_convs[1](p2)
        p = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p1 = self._fuse_topdown(laterals[0], p, stage_idx=0)
        p1 = self.fpn_convs[0](p1)
        seg_logits = self.seg_head(p1)
        edge_logits = None
        if self.use_edge_head:
            edge_logits = self.edge_head(p1)
        aux_logits = []
        if self.use_deep_supervision:
            aux_logits = [self.aux_heads[0](p2), self.aux_heads[1](p3), self.aux_heads[2](p4)]
        return (seg_logits, edge_logits, aux_logits)

@MODELS.register_module()
class SiameseMambaVision(BaseModule):
    """
    Siamese MambaVision for Change Detection

    Architecture:
        1. Siamese Backbone (shared weights): Extract multi-scale features
        2. Difference Neck: Compute and fuse differences at each scale
           - Option 1: DifferenceModule (baseline)
           - Option 2: GatedDifferenceModule (with change-aware gate, recommended)
        3. FPN Decoder: Gradually upsample and predict change map
    """

    def __init__(self, backbone_cfg=dict(dim=80, in_dim=32, depths=(1, 3, 8, 4), window_size=(8, 8, 64, 32), mlp_ratio=4.0, num_heads=(2, 4, 8, 16), drop_path_rate=0.3, out_indices=(0, 1, 2, 3)), neck_out_channels=128, min_neck_channels=128, align_mode='pre', use_gated_diff=True, diff_module_type=None, gate_type='difference', gate_channels='spatial', gate_arrangement='parallel', use_diff_in_gate=True, use_product_feature=True, norm_cfg=None, fpn_dim=128, num_classes=2, decoder_fuse_mode='add', use_deep_supervision=False, aux_loss_weights=(0.1, 0.1, 0.1), use_edge_supervision=True, edge_width=3, lambda_edge=0.2, edge_pos_weight=5.0, class_weight_bg=1.0, class_weight_change=5.0, lambda_dice=0.5, pretrained=None, allow_partial_pretrained=False, test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)), data_preprocessor=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.data_preprocessor = data_preprocessor
        self.test_cfg = test_cfg
        self.backbone = MambaVisionBackbone(**backbone_cfg)
        backbone_dims = self.backbone.dims
        if isinstance(neck_out_channels, str) and neck_out_channels.lower() in ('auto_half', 'adaptive'):
            stage_out_channels = [max(int(min_neck_channels), int(c // 2)) for c in backbone_dims]
        elif isinstance(neck_out_channels, (list, tuple)):
            if len(neck_out_channels) != len(backbone_dims):
                raise ValueError(f'neck_out_channels length ({len(neck_out_channels)}) must match number of backbone stages ({len(backbone_dims)})')
            stage_out_channels = [int(c) for c in neck_out_channels]
        else:
            stage_out_channels = [int(neck_out_channels)] * len(backbone_dims)
        self.stage_out_channels = stage_out_channels
        if diff_module_type is None:
            diff_module_type = 'gated' if use_gated_diff else 'baseline'
        if diff_module_type not in ('baseline', 'gated'):
            raise ValueError(f"Unsupported diff_module_type={diff_module_type}, expected one of ('baseline', 'gated')")
        self.diff_module_type = diff_module_type
        self.use_gated_diff = diff_module_type == 'gated'
        if diff_module_type == 'gated':
            print(f'\n Using GatedDifferenceModule (gate_type={gate_type}, gate_channels={gate_channels}) for change-aware gating')
            self.diff_modules = nn.ModuleList([GatedDifferenceModule(backbone_dims[i], stage_out_channels[i], gate_type=gate_type, gate_channels=gate_channels, gate_arrangement=gate_arrangement, norm_cfg=norm_cfg, align_mode=align_mode, use_diff_in_gate=use_diff_in_gate, use_product_feature=use_product_feature) for i in range(len(backbone_dims))])
        else:
            print('\n Using DifferenceModule (baseline)')
            self.diff_modules = nn.ModuleList([DifferenceModule(backbone_dims[i], stage_out_channels[i], norm_cfg=norm_cfg, align_mode=align_mode, use_product_feature=use_product_feature) for i in range(len(backbone_dims))])
        self.use_edge_supervision = use_edge_supervision
        self.decoder = FPNDecoder(in_channels=stage_out_channels, fpn_dim=fpn_dim, num_classes=num_classes, use_edge_head=use_edge_supervision, norm_cfg=norm_cfg, fuse_mode=decoder_fuse_mode, use_deep_supervision=use_deep_supervision)
        self.use_deep_supervision = use_deep_supervision
        if len(aux_loss_weights) != 3:
            raise ValueError('aux_loss_weights must be a 3-tuple for [P2, P3, P4]')
        self.aux_loss_weights = tuple((float(w) for w in aux_loss_weights))
        self.edge_width = edge_width
        self.lambda_edge = lambda_edge
        self.edge_pos_weight = edge_pos_weight
        self.class_weight_bg = float(class_weight_bg)
        self.class_weight_change = float(class_weight_change)
        self.lambda_dice = float(lambda_dice)
        self.allow_partial_pretrained = bool(allow_partial_pretrained)
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def train(self, mode=True):
        """Optionally keep the frozen predictor deterministic during scalar fits."""
        super().train(mode)
        return self

    def _refine_seg_logits(self, seg_logits, edge_logits):
        return seg_logits

    def load_pretrained(self, ckpt_path):
        """Load pretrained backbone weights from MambaVision checkpoint"""
        import torch
        import os
        print('\n' + '=' * 70)
        print('Loading Pretrained Weights')
        print('=' * 70)
        print(f'Checkpoint path: {ckpt_path}')
        if not os.path.exists(ckpt_path):
            print(f'?ERROR: Checkpoint file not found at {ckpt_path}')
            print('Please download the pretrained weights first.')
            return
        try:
            print(f'Loading checkpoint...')
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("?Found 'state_dict' in checkpoint")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("?Found 'model' in checkpoint")
            else:
                state_dict = checkpoint
                print('?Using checkpoint directly as state_dict')
            print(f'Total keys in checkpoint: {len(state_dict)}')
            backbone_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_k = k.replace('backbone.', '')
                    backbone_dict[new_k] = v
                elif not k.startswith(('head.', 'norm.', 'avgpool.', 'fc.', 'classifier.')):
                    backbone_dict[k] = v
            print(f'Filtered backbone keys: {len(backbone_dict)}')
            msg = self.backbone.load_state_dict(backbone_dict, strict=False)
            print('\n' + '-' * 70)
            print('Loading Report:')
            print('-' * 70)
            backbone_params = set(self.backbone.state_dict().keys())
            loaded_params = backbone_params - set(msg.missing_keys)
            print(f'?Successfully loaded: {len(loaded_params)}/{len(backbone_params)} parameters')
            print(f'  - Missing keys: {len(msg.missing_keys)}')
            print(f'  - Unexpected keys: {len(msg.unexpected_keys)}')
            if len(msg.missing_keys) > 0:
                print('\n? Missing keys (first 10):')
                for key in list(msg.missing_keys)[:10]:
                    print(f'  - {key}')
                if len(msg.missing_keys) > 10:
                    print(f'  ... and {len(msg.missing_keys) - 10} more')
            if len(msg.unexpected_keys) > 0:
                print('\n? Unexpected keys (first 10):')
                for key in list(msg.unexpected_keys)[:10]:
                    print(f'  - {key}')
                if len(msg.unexpected_keys) > 10:
                    print(f'  ... and {len(msg.unexpected_keys) - 10} more')
            load_percentage = len(loaded_params) / len(backbone_params) * 100
            print(f'\n Loading rate: {load_percentage:.1f}%')
            if load_percentage > 90:
                print('?Pretrained weights loaded successfully!')
            elif load_percentage > 50:
                print('? Partial loading: Some parameters are missing')
            else:
                print('?WARNING: Most parameters are missing!')
            if (msg.missing_keys or msg.unexpected_keys) and (not self.allow_partial_pretrained):
                raise RuntimeError(f'Pretrained backbone does not exactly match the configured architecture: {len(msg.missing_keys)} missing and {len(msg.unexpected_keys)} unexpected keys. Set allow_partial_pretrained=True only for an intentional partial-transfer experiment.')
            print('=' * 70 + '\n')
        except Exception as e:
            print(f'\n?ERROR: Failed to load pretrained weights')
            print(f'Error message: {e}')
            import traceback
            traceback.print_exc()
            print('=' * 70 + '\n')
            raise

    def forward(self, img_t1, img_t2):
        """
        Forward pass for change detection

        Args:
            img_t1: (B, 3, H, W) - Image at time t1
            img_t2: (B, 3, H, W) - Image at time t2

        Returns:
            logits: (B, num_classes, H, W) - Change prediction logits
        """
        ref_param = next(self.backbone.parameters())
        model_dtype = ref_param.dtype
        model_device = ref_param.device
        if img_t1.dtype != model_dtype:
            img_t1 = img_t1.to(dtype=model_dtype)
        if img_t1.device != model_device:
            img_t1 = img_t1.to(device=model_device)
        if img_t2.dtype != model_dtype:
            img_t2 = img_t2.to(dtype=model_dtype)
        if img_t2.device != model_device:
            img_t2 = img_t2.to(device=model_device)
        features_t1 = self.backbone(img_t1)
        features_t2 = self.backbone(img_t2)
        diff_features = []
        for i, diff_module in enumerate(self.diff_modules):
            diff_feat = diff_module(features_t1[i], features_t2[i])
            diff_features.append(diff_feat)
        seg_logits_low, edge_logits_low, aux_logits_low = self.decoder(diff_features)
        seg_logits = F.interpolate(seg_logits_low, size=(img_t1.shape[2], img_t1.shape[3]), mode='bilinear', align_corners=False)
        edge_logits = None
        if edge_logits_low is not None:
            edge_logits = F.interpolate(edge_logits_low, size=(img_t1.shape[2], img_t1.shape[3]), mode='bilinear', align_corners=False)
        seg_logits = self._refine_seg_logits(seg_logits, edge_logits)
        aux_logits = []
        for aux_logit in aux_logits_low:
            aux_logits.append(F.interpolate(aux_logit, size=(img_t1.shape[2], img_t1.shape[3]), mode='bilinear', align_corners=False))
        return (seg_logits, edge_logits, aux_logits)

    def extract_feat(self, img):
        """Extract features (for compatibility with mmseg)"""
        return self.backbone(img)

    def loss(self, inputs, data_samples):
        """
        Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor or dict): Input data, can be:
                - Tensor: concatenated images (B, 6, H, W)
                - dict: {'inputs': concatenated images}
            data_samples (list): List of data samples containing ground truth

        Returns:
            dict: A dictionary of loss components
        """
        if isinstance(inputs, dict):
            img = inputs['inputs']
        elif isinstance(inputs, list):
            img = torch.stack(inputs, dim=0) if len(inputs) > 0 else inputs[0]
        else:
            img = inputs
        if isinstance(img, list):
            img = torch.stack(img, dim=0)
        if img.shape[1] == 6:
            img_t1 = img[:, :3, :, :]
            img_t2 = img[:, 3:, :, :]
        else:
            batch_size = img.shape[0] // 2
            img_t1 = img[:batch_size]
            img_t2 = img[batch_size:]
        if self.training and 0.0 > 0.0:
            swap_mask = (torch.rand(img_t1.shape[0], device=img_t1.device) < 0.0).view(-1, 1, 1, 1)
            img_t1, img_t2 = (torch.where(swap_mask, img_t2, img_t1), torch.where(swap_mask, img_t1, img_t2))
        seg_logits, edge_logits, aux_logits = self.forward(img_t1, img_t2)
        gt_labels = torch.stack([sample.gt_sem_seg.data for sample in data_samples], dim=0)
        if gt_labels.device != seg_logits.device:
            gt_labels = gt_labels.to(device=seg_logits.device)
        gt_labels = gt_labels.squeeze(1)
        unique_before = torch.unique(gt_labels)
        padding_255_count = (gt_labels == 255).sum().item()
        total_pixels = gt_labels.numel()
        padding_ratio = padding_255_count / total_pixels if total_pixels > 0 else 0
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 3:
            print(f'\n=== Label Debug (Batch {self._debug_count}) ===')
            print(f'Unique label values: {unique_before.cpu().tolist()}')
            print(f'Label shape: {gt_labels.shape}')
            print(f'Changed pixels (==1): {(gt_labels == 1).sum().item()} / {gt_labels.numel()} ({100 * (gt_labels == 1).sum().item() / gt_labels.numel():.2f}%)')
            print(f'Background pixels (==0): {(gt_labels == 0).sum().item()} / {gt_labels.numel()} ({100 * (gt_labels == 0).sum().item() / gt_labels.numel():.2f}%)')
            print(f'255 pixels: {padding_255_count} / {total_pixels} ({padding_ratio * 100:.2f}%)')
            if padding_ratio > 0.05:
                print(f'? WARNING: High 255 ratio ({padding_ratio * 100:.2f}%) suggests unmapped change pixels!')
                print(f'    Expected: Pipeline should map 255->1 before padding')
                print(f'    Action: Will map 255->1 in loss function')
            elif padding_ratio > 0:
                print(f'?255 ratio ({padding_ratio * 100:.2f}%) looks like padding (ignore_index)')
            self._debug_count += 1
        if padding_ratio > 0.05:
            if self._debug_count <= 3:
                print(f' Mapping 255->1 (ratio={padding_ratio:.3f} > 5%, treating as change pixels)')
            gt_labels = torch.where(gt_labels == 255, torch.ones_like(gt_labels), gt_labels)
        num_classes = seg_logits.shape[1]
        invalid_mask = (gt_labels >= num_classes) & (gt_labels != 255)
        if invalid_mask.any():
            gt_labels = gt_labels.clone()
            gt_labels[invalid_mask] = 255
        class_weights = torch.tensor([self.class_weight_bg, self.class_weight_change], device=seg_logits.device, dtype=seg_logits.dtype)
        loss_ce = F.cross_entropy(seg_logits, gt_labels, weight=class_weights, ignore_index=255)
        pred_probs = F.softmax(seg_logits, dim=1)
        dice_losses = []
        for cls in range(num_classes):
            pred_cls = pred_probs[:, cls, :, :]
            gt_cls = (gt_labels == cls).float()
            valid_mask = gt_labels != 255
            pred_cls = pred_cls * valid_mask.float()
            gt_cls = gt_cls * valid_mask.float()
            intersection = (pred_cls * gt_cls).sum()
            union = pred_cls.sum() + gt_cls.sum()
            dice = (2.0 * intersection + 1e-05) / (union + 1e-05)
            dice_losses.append(1.0 - dice)
        loss_dice = sum(dice_losses) / len(dice_losses)
        loss_aux = None
        if self.use_deep_supervision and len(aux_logits) > 0:
            aux_losses = []
            for i, aux_logit in enumerate(aux_logits):
                weight = self.aux_loss_weights[i] if i < len(self.aux_loss_weights) else 0.0
                if weight <= 0:
                    continue
                aux_ce = F.cross_entropy(aux_logit, gt_labels, weight=class_weights, ignore_index=255)
                aux_losses.append(aux_ce * weight)
            if len(aux_losses) > 0:
                loss_aux = sum(aux_losses)
        loss_edge = None
        if edge_logits is not None:
            edge_gt = mask_to_edge(gt_labels, edge_width=self.edge_width, ignore_index=255)
            loss_edge = edge_bce_loss(edge_logits, edge_gt, pos_weight=self.edge_pos_weight, ignore_index=255)
        losses = dict(loss_ce=loss_ce, loss_dice=loss_dice * self.lambda_dice)
        if loss_aux is not None:
            losses['loss_aux'] = loss_aux
        if loss_edge is not None:
            losses['loss_edge'] = self.lambda_edge * loss_edge
        return losses

    def predict(self, inputs, data_samples, mode='tensor'):
        """
        Predict results from a batch of inputs and data samples.

        Args:
            inputs (Tensor, dict, or list): Input data
            data_samples (list): List of data samples
            mode (str): Return mode, 'tensor' or 'predict'

        Returns:
            list: Prediction results
        """
        if isinstance(inputs, dict):
            img = inputs['inputs']
        elif isinstance(inputs, list):
            img = torch.stack(inputs, dim=0) if len(inputs) > 0 else inputs[0]
        else:
            img = inputs
        if isinstance(img, list):
            img = torch.stack(img, dim=0)
        if img.shape[1] == 6:
            img_t1 = img[:, :3, :, :]
            img_t2 = img[:, 3:, :, :]
        else:
            batch_size = img.shape[0] // 2
            img_t1 = img[:batch_size]
            img_t2 = img[batch_size:]
        if self.test_cfg is not None and self.test_cfg.get('mode') == 'slide':
            seg_logits = self._slide_inference(img_t1, img_t2)
        else:
            seg_logits, _, _ = self.forward(img_t1, img_t2)
        if mode == 'tensor':
            return seg_logits
        preds = seg_logits.argmax(dim=1)
        results = []
        for i, data_sample in enumerate(data_samples):
            result = data_sample
            pred_data = PixelData()
            pred_data.data = preds[i:i + 1]
            result.pred_sem_seg = pred_data
            results.append(result)
        return results

    def _slide_inference(self, img_t1, img_t2):
        """
        Sliding window inference to avoid OOM on large images.
        Aggregates logits by averaging overlapping patches.
        """
        crop_h, crop_w = self.test_cfg['crop_size']
        stride_h, stride_w = self.test_cfg['stride']
        blend_mode = self.test_cfg.get('blend', 'uniform')
        blend_sigma = self.test_cfg.get('blend_sigma', 0.5)
        blend_min_weight = self.test_cfg.get('blend_min_weight', 0.1)
        B, C, H, W = img_t1.shape
        num_classes = self.decoder.seg_head[-1].out_channels
        h_steps = max(1, math.ceil((H - crop_h) / stride_h) + 1)
        w_steps = max(1, math.ceil((W - crop_w) / stride_w) + 1)
        preds = None
        count_mat = None
        blend_weight = None
        for hi in range(h_steps):
            for wi in range(w_steps):
                y1 = min(hi * stride_h, H - crop_h)
                x1 = min(wi * stride_w, W - crop_w)
                y2 = y1 + crop_h
                x2 = x1 + crop_w
                patch_t1 = img_t1[:, :, y1:y2, x1:x2]
                patch_t2 = img_t2[:, :, y1:y2, x1:x2]
                with torch.no_grad():
                    logits, _, _ = self.forward(patch_t1, patch_t2)
                if preds is None:
                    preds = logits.new_zeros((B, num_classes, H, W))
                    count_mat = logits.new_zeros((B, 1, H, W))
                    blend_weight = sliding_blend_weight(crop_h, crop_w, mode=blend_mode, sigma=blend_sigma, min_weight=blend_min_weight, device=logits.device, dtype=logits.dtype)
                preds[:, :, y1:y2, x1:x2] += logits * blend_weight
                count_mat[:, :, y1:y2, x1:x2] += blend_weight
        preds = preds / count_mat.clamp(min=1)
        return preds

    def train_step(self, data, optim_wrapper):
        """
        Training step for MMEngine Runner.

        Args:
            data (dict): Data batch from dataloader with keys:
                - 'inputs': input images
                - 'data_samples': data samples with ground truth
            optim_wrapper: Optimizer wrapper from MMEngine

        Returns:
            dict: Loss dict with log variables
        """
        inputs = data['inputs']
        data_samples = data['data_samples']
        with optim_wrapper.optim_context(self):
            losses = self.loss(inputs, data_samples)
        if isinstance(losses, dict):
            loss_total = sum(losses.values())
            log_vars = {k: v.item() if hasattr(v, 'item') else v for k, v in losses.items()}
            log_vars['loss'] = loss_total.detach().item()
        else:
            loss_total = losses
            log_vars = {'loss': losses.item() if hasattr(losses, 'item') else losses}
        optim_wrapper.update_params(loss_total)
        return log_vars

    def val_step(self, data):
        """
        Validation step for MMEngine Runner.

        Args:
            data (dict): Data batch from dataloader

        Returns:
            list: Prediction results for evaluation
        """
        inputs = data['inputs']
        data_samples = data['data_samples']
        return self.predict(inputs, data_samples, mode='predict')
