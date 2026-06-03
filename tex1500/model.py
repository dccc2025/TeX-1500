"""UNet that maps HSI patches to normalized T/e/X targets.

The trunk is intentionally simple: a 2D UNet extracts spatial-spectral HSI
features and returns a high-resolution feature map. Three lightweight heads then
predict temperature, emissivity, and texture. The emissivity head is low-rank in
the spectral axis, so a small per-pixel code can reconstruct smooth spectra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _tuple_int(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def default_group_count(channels: int, max_groups: int = 32) -> int:
    for groups in (max_groups, 24, 16, 12, 8, 6, 4, 3, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def make_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(default_group_count(channels), channels)


def maybe_checkpoint(module: nn.Module, x: Tensor, enabled: bool) -> Tensor:
    if enabled and torch.is_grad_enabled():
        return checkpoint(module, x, use_reentrant=False)
    return module(x)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            make_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResBlock2d(nn.Module):
    def __init__(self, channels: int, *, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.norm1 = make_norm(channels)
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.norm2 = make_norm(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(F.silu(self.norm2(h), inplace=True))
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, depth: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = [ConvNormAct(in_channels, out_channels, stride=2)]
        blocks.extend(ResBlock2d(out_channels) for _ in range(depth))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, depth: int) -> None:
        super().__init__()
        self.up = ConvNormAct(in_channels, out_channels)
        self.fuse = ConvNormAct(out_channels + skip_channels, out_channels, kernel_size=1, padding=0)
        self.blocks = nn.Sequential(*[ResBlock2d(out_channels) for _ in range(depth)])

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.blocks(self.fuse(x))


class SpectralResidual1d(nn.Module):
    """Chunked residual 1D conv over the spectral axis."""

    def __init__(
        self,
        num_bands: int,
        *,
        hidden_channels: int = 16,
        kernel_size: int = 5,
        chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.num_bands = int(num_bands)
        self.chunk_size = int(chunk_size)
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(num_bands)
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, padding=padding),
            make_norm(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, groups=hidden_channels),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def _forward_flat(self, x_flat: Tensor) -> Tensor:
        return self.net(self.norm(x_flat).unsqueeze(1)).squeeze(1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.num_bands:
            raise ValueError(f"x must be [B,{self.num_bands},H,W], got {tuple(x.shape)}")
        b, c, h, w = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        if self.chunk_size > 0 and flat.shape[0] > self.chunk_size:
            out = torch.cat([self._forward_flat(chunk) for chunk in flat.split(self.chunk_size, dim=0)], dim=0)
        else:
            out = self._forward_flat(flat)
        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + out


class WavelengthBasis(nn.Module):
    """Map wavelengths to low-rank spectral basis vectors."""

    def __init__(
        self,
        code_dim: int,
        *,
        hidden_dim: int = 128,
        num_frequencies: int = 16,
        min_um: float = 6.0,
        max_um: float = 14.0,
    ) -> None:
        super().__init__()
        if max_um <= min_um:
            raise ValueError("max_um must be larger than min_um")
        self.min_um = float(min_um)
        self.max_um = float(max_um)
        freqs = torch.logspace(0.0, math.log2(32.0), steps=num_frequencies, base=2.0)
        self.register_buffer("frequencies", freqs, persistent=False)
        in_dim = 1 + 2 * num_frequencies
        self.basis = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, code_dim),
        )
        self.bias = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def _features(self, wavelength_um: Tensor) -> Tensor:
        x = 2.0 * (wavelength_um - self.min_um) / (self.max_um - self.min_um) - 1.0
        angles = math.pi * x.unsqueeze(-1) * self.frequencies
        return torch.cat([x.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, wavelength_um: Tensor, *, batch_size: int) -> tuple[Tensor, Tensor]:
        if wavelength_um.ndim == 1:
            wavelength_um = wavelength_um.unsqueeze(0).expand(batch_size, -1)
        elif wavelength_um.ndim != 2:
            raise ValueError(f"wavelength_um must be [C] or [B,C], got {tuple(wavelength_um.shape)}")
        if wavelength_um.shape[0] != batch_size:
            if wavelength_um.shape[0] == 1:
                wavelength_um = wavelength_um.expand(batch_size, -1)
            else:
                raise ValueError(f"wavelength batch mismatch: {wavelength_um.shape[0]} vs {batch_size}")
        feats = self._features(wavelength_um.to(dtype=torch.float32))
        return self.basis(feats), self.bias(feats).squeeze(-1)


class WavelengthConditionedProjection(nn.Module):
    """Project HSI bands into stem features and add learnable wavelength PE."""

    def __init__(
        self,
        num_bands: int,
        out_channels: int,
        *,
        hidden_dim: int = 128,
        num_frequencies: int = 16,
        min_um: float = 6.0,
        max_um: float = 14.0,
    ) -> None:
        super().__init__()
        if num_bands <= 0 or out_channels <= 0:
            raise ValueError("num_bands and out_channels must be positive")
        if max_um <= min_um:
            raise ValueError("max_um must be larger than min_um")
        self.num_bands = int(num_bands)
        self.out_channels = int(out_channels)
        self.min_um = float(min_um)
        self.max_um = float(max_um)
        freqs = torch.logspace(0.0, math.log2(32.0), steps=num_frequencies, base=2.0)
        self.register_buffer("frequencies", freqs, persistent=False)
        self.register_buffer(
            "default_wavelength_um",
            torch.linspace(min_um, max_um, steps=num_bands),
            persistent=False,
        )
        in_dim = 1 + 2 * num_frequencies
        self.weight_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_channels),
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_channels),
        )
        self.bias_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_channels),
        )

    def _features(self, wavelength_um: Tensor) -> Tensor:
        x = 2.0 * (wavelength_um - self.min_um) / (self.max_um - self.min_um) - 1.0
        angles = math.pi * x.unsqueeze(-1) * self.frequencies
        return torch.cat([x.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)

    def _prepare_wavelength(self, wavelength_um: Tensor | None, *, batch_size: int, device: torch.device) -> Tensor:
        if wavelength_um is None:
            wavelength_um = self.default_wavelength_um
        wavelength_um = wavelength_um.to(device=device, dtype=torch.float32)
        if wavelength_um.ndim == 1:
            wavelength_um = wavelength_um.unsqueeze(0).expand(batch_size, -1)
        elif wavelength_um.ndim != 2:
            raise ValueError(f"wavelength_um must be [C] or [B,C], got {tuple(wavelength_um.shape)}")
        if wavelength_um.shape[0] != batch_size:
            if wavelength_um.shape[0] == 1:
                wavelength_um = wavelength_um.expand(batch_size, -1)
            else:
                raise ValueError(f"wavelength batch mismatch: {wavelength_um.shape[0]} vs {batch_size}")
        if wavelength_um.shape[1] != self.num_bands:
            raise ValueError(f"wavelength has {wavelength_um.shape[1]} bands, expected {self.num_bands}")
        return wavelength_um

    def forward(self, values: Tensor, wavelength_um: Tensor | None = None) -> Tensor:
        if values.ndim != 4:
            raise ValueError(f"values must be [B,C,H,W], got {tuple(values.shape)}")
        if values.shape[1] != self.num_bands:
            raise ValueError(f"values has {values.shape[1]} bands, expected {self.num_bands}")
        batch = int(values.shape[0])
        wavelength_um = self._prepare_wavelength(wavelength_um, batch_size=batch, device=values.device)
        feats = self._features(wavelength_um)
        weights = self.weight_mlp(feats) / math.sqrt(float(self.num_bands))
        position = self.position_mlp(feats).mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        bias = self.bias_mlp(feats.mean(dim=1)).unsqueeze(-1).unsqueeze(-1)
        return torch.einsum("bchw,bcd->bdhw", values, weights) + position + bias


class SpectralBasisHead(nn.Module):
    """Predict emissivity spectra from compact per-pixel codes."""

    def __init__(
        self,
        in_channels: int,
        num_bands: int,
        *,
        code_dim: int = 16,
        hidden_channels: int = 128,
        wavelength_hidden_dim: int = 128,
        wavelength_min_um: float = 6.0,
        wavelength_max_um: float = 14.0,
        spectral_refine_depth: int = 0,
        bound_output: bool = True,
    ) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.code_dim = int(code_dim)
        self.bound_output = bool(bound_output)
        self.code = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels, kernel_size=1, padding=0),
            nn.Conv2d(hidden_channels, code_dim, kernel_size=1),
        )
        self.wavelength_basis = WavelengthBasis(
            code_dim,
            hidden_dim=wavelength_hidden_dim,
            min_um=wavelength_min_um,
            max_um=wavelength_max_um,
        )
        self.register_buffer(
            "default_wavelength_um",
            torch.linspace(wavelength_min_um, wavelength_max_um, steps=num_bands),
            persistent=False,
        )
        self.refine = nn.Sequential(
            *[SpectralResidual1d(num_bands, hidden_channels=16, kernel_size=5) for _ in range(spectral_refine_depth)]
        )
        nn.init.zeros_(self.code[-1].weight)
        nn.init.zeros_(self.code[-1].bias)
        nn.init.zeros_(self.wavelength_basis.bias[-1].weight)
        nn.init.zeros_(self.wavelength_basis.bias[-1].bias)

    def forward(self, features: Tensor, wavelength_um: Tensor | None = None) -> tuple[Tensor, Tensor]:
        code = self.code(features)
        batch = int(features.shape[0])
        if wavelength_um is None:
            wavelength_um = self.default_wavelength_um.to(device=features.device)
        basis, bias = self.wavelength_basis(wavelength_um.to(device=features.device), batch_size=batch)
        if basis.shape[1] != self.num_bands:
            raise ValueError(f"basis has {basis.shape[1]} bands, expected {self.num_bands}")
        logits = torch.einsum("bkhw,bck->bchw", code, basis / math.sqrt(float(self.code_dim)))
        logits = logits + bias.unsqueeze(-1).unsqueeze(-1)
        logits = self.refine(logits)
        if self.bound_output:
            logits = torch.sigmoid(logits)
        return logits, code


class ScalarHead(nn.Module):
    def __init__(self, in_channels: int, *, hidden_channels: int = 128, bound_output: bool = True) -> None:
        super().__init__()
        self.bound_output = bool(bound_output)
        self.net = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels, kernel_size=3),
            ResBlock2d(hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        if self.bound_output:
            x = torch.sigmoid(x)
        return x


@dataclass(frozen=True)
class TeXUNetConfig:
    num_bands: int = 64
    channels: tuple[int, ...] = (64, 128, 256, 384, 512)
    depths: tuple[int, ...] = (2, 2, 2, 2, 2)
    trunk_channels: int = 256
    head_hidden_channels: int = 192
    e_code_dim: int = 32
    e_wavelength_hidden_dim: int = 384
    hsi_wavelength_encoding: bool = True
    hsi_wavelength_hidden_dim: int = 128
    wavelength_min_um: float = 6.0
    wavelength_max_um: float = 14.0
    spectral_refine_depth: int = 0
    bound_outputs: bool = True
    activation_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if len(self.channels) < 2:
            raise ValueError("channels must contain at least stem and bottleneck channels")
        if len(self.channels) != len(self.depths):
            raise ValueError("channels and depths must have the same length")
        if self.trunk_channels <= 0 or self.e_code_dim <= 0:
            raise ValueError("trunk_channels and e_code_dim must be positive")
        if self.head_hidden_channels <= 0 or self.hsi_wavelength_hidden_dim <= 0:
            raise ValueError("head_hidden_channels and hsi_wavelength_hidden_dim must be positive")


class TeXUNet(nn.Module):
    """HSI -> T/e/X network.

    Args:
        hsi: normalized HSI tensor [B,C,H,W].
        wavelength_um: optional working wavelength tensor [C] or [B,C].

    Returns:
        dict with T [B,1,H,W], e [B,C,H,W], X [B,1,H,W], and e_code.
    """

    def __init__(self, config: TeXUNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or TeXUNetConfig()
        cfg = self.config
        channels = _tuple_int(cfg.channels)
        depths = _tuple_int(cfg.depths)

        self.stem_in = ConvNormAct(cfg.num_bands, channels[0], kernel_size=3)
        self.hsi_wavelength_projection = (
            WavelengthConditionedProjection(
                cfg.num_bands,
                channels[0],
                hidden_dim=cfg.hsi_wavelength_hidden_dim,
                min_um=cfg.wavelength_min_um,
                max_um=cfg.wavelength_max_um,
            )
            if cfg.hsi_wavelength_encoding
            else None
        )
        self.hsi_wavelength_norm = (
            nn.Sequential(make_norm(channels[0]), nn.SiLU(inplace=True))
            if cfg.hsi_wavelength_encoding
            else None
        )
        self.stem_blocks = nn.Sequential(*[ResBlock2d(channels[0]) for _ in range(depths[0])])
        self.downs = nn.ModuleList(
            [
                DownBlock(channels[idx - 1], channels[idx], depth=depths[idx])
                for idx in range(1, len(channels))
            ]
        )

        up_blocks: list[nn.Module] = []
        current = channels[-1]
        for idx in range(len(channels) - 2, -1, -1):
            skip_channels = channels[idx]
            up_blocks.append(UpBlock(current, skip_channels, skip_channels, depth=depths[idx]))
            current = skip_channels
        self.ups = nn.ModuleList(up_blocks)

        self.trunk = nn.Sequential(
            ConvNormAct(channels[0], cfg.trunk_channels, kernel_size=3),
            ResBlock2d(cfg.trunk_channels),
        )
        self.t_head = ScalarHead(
            cfg.trunk_channels,
            hidden_channels=cfg.head_hidden_channels,
            bound_output=cfg.bound_outputs,
        )
        self.x_head = ScalarHead(
            cfg.trunk_channels,
            hidden_channels=cfg.head_hidden_channels,
            bound_output=cfg.bound_outputs,
        )
        self.e_head = SpectralBasisHead(
            cfg.trunk_channels,
            cfg.num_bands,
            code_dim=cfg.e_code_dim,
            hidden_channels=cfg.head_hidden_channels,
            wavelength_hidden_dim=cfg.e_wavelength_hidden_dim,
            wavelength_min_um=cfg.wavelength_min_um,
            wavelength_max_um=cfg.wavelength_max_um,
            spectral_refine_depth=cfg.spectral_refine_depth,
            bound_output=cfg.bound_outputs,
        )

    def forward(
        self,
        hsi: Tensor,
        wavelength_um: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if hsi.ndim != 4:
            raise ValueError(f"hsi must be [B,C,H,W], got {tuple(hsi.shape)}")
        if hsi.shape[1] != self.config.num_bands:
            raise ValueError(f"hsi has {hsi.shape[1]} bands, expected {self.config.num_bands}")

        use_ckpt = self.config.activation_checkpoint
        x = maybe_checkpoint(self.stem_in, hsi, use_ckpt)
        if self.hsi_wavelength_projection is not None:
            hsi_pos = self.hsi_wavelength_projection(hsi, wavelength_um)
            if self.hsi_wavelength_norm is not None:
                hsi_pos = self.hsi_wavelength_norm(hsi_pos)
            x = x + hsi_pos
        x = maybe_checkpoint(self.stem_blocks, x, use_ckpt)
        skips = [x]
        for down in self.downs:
            x = maybe_checkpoint(down, x, use_ckpt)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)
        features = maybe_checkpoint(self.trunk, x, use_ckpt)
        emissivity, e_code = self.e_head(features, wavelength_um)
        return {
            "T": self.t_head(features),
            "e": emissivity,
            "X": self.x_head(features),
            "features": features,
            "e_code": e_code,
        }


def count_parameters(module: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
