# models/unet3d.py
# Minimal, solid 3D U-Net for volumetric segmentation (PyTorch)
# Works with your loaders: image [B,1,D,H,W], label [B,1,D,H,W]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

# ----------------------- building blocks -----------------------

def conv_block(in_ch: int, out_ch: int, *, norm='instance', dropout: float = 0.0):
    """
    Two 3x3x3 convs with same padding + norm + ReLU. InstanceNorm3d by default.
    """
    norm_layer = {
        'batch':   nn.BatchNorm3d,
        'instance': nn.InstanceNorm3d,
        'group':   lambda c: nn.GroupNorm(num_groups=min(8, c), num_channels=c),
        None:      None
    }[norm]

    layers = []
    # Conv 1
    layers += [nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=norm is None)]
    if norm_layer is not None:
        layers += [norm_layer(out_ch, affine=True)]
    layers += [nn.ReLU(inplace=True)]
    if dropout and dropout > 0:
        layers += [nn.Dropout3d(p=dropout)]

    # Conv 2
    layers += [nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=norm is None)]
    if norm_layer is not None:
        layers += [norm_layer(out_ch, affine=True)]
    layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)


class Down(nn.Module):
    """Downsampling step: MaxPool3d -> conv block."""
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block = conv_block(in_ch, out_ch, norm=norm, dropout=dropout)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    """Upsampling step: transposed conv -> concat skip -> conv block."""
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0, use_transpose=True):
        super().__init__()
        if use_transpose:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
            in_after_up = out_ch
        else:
            # trilinear upsample + 1x1 conv to reduce channels
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.Conv3d(in_ch, out_ch, kernel_size=1)
            )
            in_after_up = out_ch
        self.block = conv_block(in_after_up * 2, out_ch, norm=norm, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if necessary (odd dims). We assume channels-first [B,C,D,H,W].
        diffD = skip.size(2) - x.size(2)
        diffH = skip.size(3) - x.size(3)
        diffW = skip.size(4) - x.size(4)
        if diffD or diffH or diffW:
            x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                          diffH // 2, diffH - diffH // 2,
                          diffD // 2, diffD - diffD // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


# ----------------------- UNet3D model -----------------------

class UNet3D(nn.Module):
    """
    Classic 3D U-Net.
    - in_channels: 1 for your MRI volumes
    - out_channels: number of classes (C); output logits (no softmax)
    - features: base channels at each depth (e.g., [32,64,128,256,512])
    - norm: 'instance' (default), 'batch', 'group', or None
    - dropout: applied inside conv blocks (encoder & decoder)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        features: Sequence[int] = (32, 64, 128, 256, 512),
        norm: str = 'instance',
        dropout: float = 0.0,
        use_transpose: bool = True,
    ):
        super().__init__()
        assert len(features) >= 4, "Use at least 4 feature levels"

        self.enc1 = conv_block(in_channels, features[0], norm=norm, dropout=dropout/2)
        self.down1 = Down(features[0], features[1], norm=norm, dropout=dropout/2)
        self.down2 = Down(features[1], features[2], norm=norm, dropout=dropout/2)
        self.down3 = Down(features[2], features[3], norm=norm, dropout=dropout/2)

        # Optional extra depth if provided
        if len(features) == 5:
            self.down4 = Down(features[3], features[4], norm=norm, dropout=dropout)
            decoder_in = features[4]
            dec_feats = features[3]
            self.deepest = True
        else:
            decoder_in = features[3]
            dec_feats = features[2]
            self.deepest = False

        # Decoder
        self.up1 = Up(decoder_in, dec_feats, norm=norm, dropout=dropout, use_transpose=use_transpose)
        self.up2 = Up(dec_feats, features[1], norm=norm, dropout=dropout, use_transpose=use_transpose)
        self.up3 = Up(features[1], features[0], norm=norm, dropout=dropout, use_transpose=use_transpose)

        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # -> f0
        e2 = self.down1(e1)        # -> f1
        e3 = self.down2(e2)        # -> f2
        e4 = self.down3(e3)        # -> f3

        if self.deepest:
            center = self.down4(e4)     # -> f4
            x = self.up1(center, e4)
        else:
            x = self.up1(e4, e3)

        x = self.up2(x, e2)
        x = self.up3(x, e1)

        logits = self.out_conv(x)
        return logits


# ----------------------- quick self-test -----------------------

if __name__ == "__main__":
    # Sanity check on shapes
    model = UNet3D(in_channels=1, out_channels=4, features=(32,64,128,256,512), dropout=0.1)
    x = torch.randn(2, 1, 64, 128, 128)  # [B,C,D,H,W]  (D/H/W must be divisible by 8 or will be padded by Up)
    y = model(x)
    print("Input:", x.shape, "Output:", y.shape)  # Expect [2, 4, 64, 128, 128]
