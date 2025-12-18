"""Siamese UNet model built on ResNet-18 encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


class ConvBlock(nn.Module):
    """Two consecutive conv-bn-relu blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample + fusion + conv block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        fusion_channels = skip_channels * 2 + skip_channels  # pre + post + diff
        self.reduce = nn.Conv2d(in_channels + fusion_channels, out_channels, kernel_size=1)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, f_pre: torch.Tensor, f_post: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=f_pre.shape[-2:], mode="bilinear", align_corners=False)
        diff = torch.abs(f_post - f_pre)
        fusion = torch.cat([f_pre, f_post, diff], dim=1)
        x = torch.cat([x, fusion], dim=1)
        x = self.reduce(x)
        return self.conv(x)


class SiameseResNet18UNet(nn.Module):
    """Siamese UNet leveraging ResNet-18 encoder features."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 * 3, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)
        self.decoder0 = ConvBlock(32, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor):
        features = []
        x0 = self.stem(x)  # 1/2 spatial
        features.append(x0)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        features.append(x1)
        x2 = self.layer2(x1)
        features.append(x2)
        x3 = self.layer3(x2)
        features.append(x3)
        x4 = self.layer4(x3)
        features.append(x4)
        return features  # [c1, c2, c3, c4, c5]

    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor) -> torch.Tensor:
        f_pre = self.encode(pre_img)
        f_post = self.encode(post_img)

        x = torch.cat([f_pre[-1], f_post[-1], torch.abs(f_post[-1] - f_pre[-1])], dim=1)
        x = self.bottleneck(x)

        x = self.decoder4(x, f_pre[-2], f_post[-2])
        x = self.decoder3(x, f_pre[-3], f_post[-3])
        x = self.decoder2(x, f_pre[-4], f_post[-4])
        x = self.decoder1(x, f_pre[-5], f_post[-5])
        x = self.decoder0(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.final_conv(x)


__all__ = ["SiameseResNet18UNet"]
