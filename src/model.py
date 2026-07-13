"""
YOLOv3 from scratch: Darknet-53 backbone + 3-scale detection head.
Kept close to the original paper. Conv2d layers are named predictably
(backbone.stage{i}... / head.branch{i}...) so the Tucker-2 pipeline can
enumerate and skip first/last layers the same way it does for VGG/ResNet.
"""
import torch
import torch.nn as nn


def conv_bn_leaky(in_ch, out_ch, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class ResidualBlock(nn.Module):
    """1x1 reduce -> 3x3 expand, residual add. Core Darknet-53 unit."""
    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.conv1 = conv_bn_leaky(channels, half, 1)
        self.conv2 = conv_bn_leaky(half, channels, 3)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out


def make_stage(in_ch, out_ch, num_blocks):
    """Downsampling conv (stride 2) followed by `num_blocks` residual blocks."""
    layers = [conv_bn_leaky(in_ch, out_ch, 3, stride=2)]
    layers += [ResidualBlock(out_ch) for _ in range(num_blocks)]
    return nn.Sequential(*layers)


class Darknet53(nn.Module):
    """
    Backbone. Returns features at 3 scales (stride 8, 16, 32) for the
    detection head, matching the original YOLOv3 multi-scale design.
    """
    def __init__(self):
        super().__init__()
        self.stem = conv_bn_leaky(3, 32, 3)
        self.stage1 = make_stage(32, 64, 1)
        self.stage2 = make_stage(64, 128, 2)
        self.stage3 = make_stage(128, 256, 8)   # -> route to head (stride 8)
        self.stage4 = make_stage(256, 512, 8)   # -> route to head (stride 16)
        self.stage5 = make_stage(512, 1024, 4)  # -> route to head (stride 32)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        route_small = self.stage3(x)     # 256 ch, stride 8
        route_medium = self.stage4(route_small)   # 512 ch, stride 16
        route_large = self.stage5(route_medium)    # 1024 ch, stride 32
        return route_small, route_medium, route_large


class ConvSet(nn.Module):
    """5-conv block used before each detection branch (1x1/3x3 alternating)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_leaky(in_ch, out_ch, 1),
            conv_bn_leaky(out_ch, out_ch * 2, 3),
            conv_bn_leaky(out_ch * 2, out_ch, 1),
            conv_bn_leaky(out_ch, out_ch * 2, 3),
            conv_bn_leaky(out_ch * 2, out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)


class DetectionHead(nn.Module):
    def __init__(self, in_ch, num_anchors, num_classes):
        super().__init__()
        out_ch = num_anchors * (5 + num_classes)  # 5 = tx,ty,tw,th,obj
        self.conv = conv_bn_leaky(in_ch, in_ch * 2, 3)
        self.pred = nn.Conv2d(in_ch * 2, out_ch, 1)

    def forward(self, x):
        return self.pred(self.conv(x))


class YOLOv3(nn.Module):
    def __init__(self, num_classes=20, anchors_per_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = anchors_per_scale
        self.backbone = Darknet53()

        # large scale (stride 32) - detects big objects
        self.set_large = ConvSet(1024, 512)
        self.head_large = DetectionHead(512, anchors_per_scale, num_classes)

        # medium scale (stride 16)
        self.reduce_medium = conv_bn_leaky(512, 256, 1)
        self.set_medium = ConvSet(256 + 512, 256)
        self.head_medium = DetectionHead(256, anchors_per_scale, num_classes)

        # small scale (stride 8)
        self.reduce_small = conv_bn_leaky(256, 128, 1)
        self.set_small = ConvSet(128 + 256, 128)
        self.head_small = DetectionHead(128, anchors_per_scale, num_classes)

    def forward(self, x):
        route_small, route_medium, route_large = self.backbone(x)

        x = self.set_large(route_large)
        out_large = self.head_large(x)

        x = self.reduce_medium(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, route_medium], dim=1)
        x = self.set_medium(x)
        out_medium = self.head_medium(x)

        x = self.reduce_small(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, route_small], dim=1)
        x = self.set_small(x)
        out_small = self.head_small(x)

        # each: (B, anchors*(5+C), H, W) at strides 32/16/8
        return out_large, out_medium, out_small


def count_conv_params(model):
    """Utility used later by the decomposition + analysis notebooks."""
    total, conv = 0, 0
    for m in model.modules():
        for p in m.parameters(recurse=False):
            total += p.numel()
        if isinstance(m, nn.Conv2d):
            conv += sum(p.numel() for p in m.parameters())
    return total, conv


if __name__ == "__main__":
    m = YOLOv3(num_classes=20)
    x = torch.randn(2, 3, 416, 416)
    outs = m(x)
    for o in outs:
        print(o.shape)
    total, conv = count_conv_params(m)
    print(f"total params: {total:,}  conv params: {conv:,} ({conv/total:.1%})")
