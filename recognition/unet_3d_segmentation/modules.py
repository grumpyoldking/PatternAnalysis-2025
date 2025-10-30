import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct3d(nn.Module):
    """
    Convenience block: 3D Conv -> GroupNorm -> Activation.
    Keeps tensor shape (except for stride/padding effects) and improves training stability.
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=8, act="silu"):
        super().__init__()
        # Bias is unnecessary because GroupNorm has affine parameters
        self.conv = nn.Conv3d(in_ch, out_ch, k, s, p, bias=False)
        # Use GroupNorm for small batch sizes; cap groups at out_ch to avoid invalid configs
        self.norm = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        # Flexible nonlinearity selection
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "lrelu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Standard Conv -> Norm -> Act pipeline
        return self.act(self.norm(self.conv(x)))


class ResBlock3d(nn.Module):
    """
    Two ConvNormAct3d blocks with a residual (skip) connection.
    Optional dropout between the two convs to regularize features.
    """
    def __init__(self, in_ch, out_ch, mid_ch=None, dropout=0.0, groups=8, act="silu"):
        super().__init__()
        mid_ch = mid_ch or out_ch
        # First conv block (change channels to mid_ch if specified)
        self.conv1 = ConvNormAct3d(in_ch, mid_ch, k=3, s=1, p=1, groups=groups, act=act)
        # Optional spatial dropout for regularization
        self.drop = nn.Dropout3d(p=dropout) if dropout and dropout > 0 else nn.Identity()
        # Second conv block maps to out_ch
        self.conv2 = ConvNormAct3d(mid_ch, out_ch, k=3, s=1, p=1, groups=groups, act=act)
        # Projection for skip path if channel dims change
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        # Main path
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        # Residual addition (identity or 1x1 projection)
        return y + self.proj(x)


class SCSE3d(nn.Module):
    """
    Concurrent Spatial & Channel Squeeze-Excitation for 3D features.
    - Channel SE: global pooling + bottleneck MLP to reweight channels.
    - Spatial SE: 1x1 conv to produce a spatial attention map.
    Final output combines both (elementwise).
    """
    def __init__(self, ch, r=16):
        super().__init__()
        red = max(ch // r, 1)
        # Channel SE branch
        self.cse_avg = nn.AdaptiveAvgPool3d(1)     # [B,C,D,H,W] -> [B,C,1,1,1]
        self.cse_fc = nn.Sequential(
            nn.Conv3d(ch, red, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(red, ch, 1, bias=True),
            nn.Sigmoid(),                          # channel weights in [0,1]
        )
        # Spatial SE branch
        self.sse = nn.Sequential(
            nn.Conv3d(ch, 1, 1, bias=True),        # compress channels to 1
            nn.Sigmoid()                           # spatial weights in [0,1]
        )

    def forward(self, x):
        # Channel weights (broadcast along spatial dims)
        c = self.cse_fc(self.cse_avg(x))
        # Spatial weights (broadcast along channel dim)
        s = self.sse(x)
        # Combine: reweight by both channel and spatial attentions
        return x * c + x * s


class AttGate3d(nn.Module):
    """
    Attention gate used on skip connections (as in Attention U-Net).
    Conditions the skip (encoder) features with a decoder gating signal to suppress irrelevant regions.
    """
    def __init__(self, in_skip, in_g, inter):
        super().__init__()
        # Linear projections to a common intermediate feature space
        self.theta = nn.Conv3d(in_skip, inter, 1, bias=False)
        self.phi   = nn.Conv3d(in_g,    inter, 1, bias=False)
        self.act   = nn.ReLU(inplace=True)
        # Attention mask -> 1 channel with sigmoid
        self.psi   = nn.Sequential(nn.Conv3d(inter, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, skip, g):
        # Upsample gating signal if spatial sizes differ
        if skip.shape[2:] != g.shape[2:]:
            g = F.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        # Compute attention coefficients
        a = self.act(self.theta(skip) + self.phi(g))
        att = self.psi(a)
        # Modulate skip features
        return skip * att


class DownBlock3d(nn.Module):
    """
    Encoder block:
      Strided Conv (downsample by 2) -> Norm+Act -> Residual Block -> SCSE attention.
    """
    def __init__(self, in_ch, out_ch, dropout=0.0, groups=8, act="silu"):
        super().__init__()
        # Learnable downsampling via strided conv
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act  = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)
        # Local feature refinement + residual learning
        self.block = ResBlock3d(out_ch, out_ch, dropout=dropout, groups=groups, act=act)
        # Channel+spatial attention to emphasize salient features
        self.scse  = SCSE3d(out_ch)

    def forward(self, x):
        x = self.act(self.norm(self.down(x)))  # downsample + normalize + nonlinearity
        x = self.block(x)                      # residual refinement
        x = self.scse(x)                       # attention reweighting
        return x


class UpBlock3d(nn.Module):
    """
    Decoder block:
      Upsample (interpolate or deconv) -> channel reduce (if needed) ->
      Attention-gated skip fusion -> Residual fuse -> SCSE.
    """
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0, groups=8, act="silu", up_mode="trilinear"):
        super().__init__()
        self.up_mode = up_mode
        if up_mode == "deconv":
            # Learnable upsampling with transposed conv
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
            up_out = out_ch
        else:
            # Fixed upsampling followed by 1x1 conv to set channels
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            self.reduce = nn.Conv3d(in_ch, out_ch, 1, bias=False)
            up_out = out_ch

        # Gate the encoder skip with the decoder signal
        self.gate = AttGate3d(skip_ch, up_out, inter=max(out_ch // 2, 8))
        # Fuse gated skip and upsampled decoder features
        self.fuse = ResBlock3d(out_ch + skip_ch, out_ch, dropout=dropout, groups=groups, act=act)
        # Attention after fusion
        self.scse = SCSE3d(out_ch)

    def forward(self, x, skip):
        # Upsample decoder features
        if self.up_mode == "deconv":
            x = self.up(x)
        else:
            x = self.reduce(self.up(x))
        # Gate the skip connection and concatenate
        skip = self.gate(skip, x)
        x = torch.cat([x, skip], dim=1)
        # Residual fusion + attention
        x = self.fuse(x)
        x = self.scse(x)
        return x


class ImprovedUNet3D(nn.Module):
    """
    Encoder-decoder UNet backbone with:
      - Residual blocks (stability/gradient flow)
      - SCSE attention (channel + spatial emphasis)
      - Attention-gated skip connections (suppress irrelevant encoder features)
      - Learnable downsampling and flexible upsampling
      - Optional deep supervision (extra outputs at decoder stages)
    """
    def __init__(
        self,
        in_channels: int = 1,                  # input channel count (e.g., 1 for single-modality MRI)
        out_channels: int = 2,                 # number of segmentation classes (logits)
        features=(32, 64, 128, 256, 512),      # channel widths per stage (depth = len(features))
        dropout: float = 0.1,                  # dropout inside residual blocks
        groups: int = 8,                       # GroupNorm groups
        act: str = "silu",                     # activation kind for conv blocks
        up_mode: str = "trilinear",            # 'trilinear' or 'deconv'
        deep_supervision: bool = False,        # if True, returns (logits, aux_logits_list)
    ):
        super().__init__()
        assert len(features) >= 4, "Use at least 4 stages for 3D volumes."

        self.deep_supervision = deep_supervision
        chans = list(features)

        # Initial feature extraction (no downsampling here)
        self.stem = ResBlock3d(in_channels, chans[0], dropout=dropout, groups=groups, act=act)
        self.stem_scse = SCSE3d(chans[0])

        # Encoder pathway: repeated downsampling + residual refinement
        self.enc = nn.ModuleList()
        for i in range(len(chans) - 1):
            self.enc.append(DownBlock3d(chans[i], chans[i+1], dropout=dropout, groups=groups, act=act))

        # Decoder pathway: upsample and fuse with gated skips, mirroring encoder
        self.dec = nn.ModuleList()
        for i in reversed(range(len(chans) - 1)):
            in_ch  = chans[i+1]   # input from deeper level
            skip_ch= chans[i]     # skip from encoder level i
            out_ch = chans[i]     # output channels match skip level
            self.dec.append(UpBlock3d(in_ch, skip_ch, out_ch, dropout=dropout, groups=groups, act=act, up_mode=up_mode))

        # Final classifier head maps to logits per class
        self.head = nn.Conv3d(chans[0], out_channels, 1, bias=True)

        # Optional auxiliary heads for deep supervision (earlier decoder stages)
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv3d(chans[i], out_channels, 1, bias=True) for i in range(1, len(chans))
            ])

    def forward(self, x):
        # ----- Encoder -----
        s0 = self.stem_scse(self.stem(x))  # stem + attention
        feats = [s0]                       # collect for skip connections
        y = s0
        for down in self.enc:
            y = down(y)
            feats.append(y)               # encoder features at each resolution

        # ----- Decoder -----
        aux = []                          # store intermediate maps for deep supervision
        for i, up in enumerate(self.dec):
            skip = feats[-(i+2)]          # pick matching skip (reverse order)
            y = up(y, skip)               # upsample + fuse with gated skip
            if self.deep_supervision and i < len(self.dec) - 1:
                aux.append(y)             # save features before final head

        # Final logits at full decoder resolution
        logits = self.head(y)

        # Standard mode: return only primary logits
        if not self.deep_supervision:
            return logits

        # Deep supervision mode: upsample intermediate features to input scale and map to logits
        aux_logits = []
        for i, y_i in enumerate(aux):
            scale = 2 ** (i + 1)  # each earlier stage is 2x smaller along D/H/W
            up = F.interpolate(y_i, scale_factor=scale, mode="trilinear", align_corners=False)
            aux_logits.append(self.aux_heads[-(i+2)](up))
        return logits, aux_logits
