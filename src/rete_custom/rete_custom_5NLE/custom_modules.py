#https://www.nature.com/articles/s41598-024-54428-8

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. Attenzione SimAM
# ----------------------------------------------------------------------
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)


# ----------------------------------------------------------------------
# 2. Modulo LEF (Filtro a basse frequenze)
# ----------------------------------------------------------------------
class LEF(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1)
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((s, s)) for s in [1, 2, 3, 6]
        ])

    def forward(self, x):
        x_proj = self.conv(x)
        h, w = x_proj.shape[-2:]
        outs = [x_proj]
        for pool in self.pools:
            pooled = pool(x_proj)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            outs.append(upsampled)
        return torch.cat(outs, dim=1)


# ----------------------------------------------------------------------
# 3. Modulo FEM (Feature Enhancement Module)
# ----------------------------------------------------------------------
class FEM(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in, max(1, c_in // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, c_in // 4), c_in, 1),
            nn.Sigmoid()
        )
        self.conv_out = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x_low, x_orig):
        concat_feat = torch.cat([x_low, x_orig], dim=1)
        weights = self.ca(concat_feat)
        weighted_feat = concat_feat * weights
        out = self.conv_out(weighted_feat)
        if x_orig.shape == out.shape:
            out = out + x_orig
        return out


# ----------------------------------------------------------------------
# 4. Moduli C2fLEFEM e AMC2fLEFEM
# ----------------------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = nn.Sequential(nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.SiLU())
        self.cv2 = nn.Sequential(nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.SiLU())
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2fLEFEM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        self.lef = LEF(c1, max(1, c1 // 2))
        lef_channels = max(1, c1 // 2) * 5 + c1
        self.fem = FEM(lef_channels, c1)
        
        c_ = int(c2 * 0.5)
        self.c2f_conv = nn.Sequential(nn.Conv2d(c1, 2 * c_, 1), nn.BatchNorm2d(2 * c_), nn.SiLU())
        self.bottlenecks = nn.ModuleList([Bottleneck(c_, c_, shortcut) for _ in range(n)])
        self.cv2 = nn.Sequential(nn.Conv2d((2 + n) * c_, c2, 1), nn.BatchNorm2d(c2), nn.SiLU())

    def forward(self, x):
        x_low = self.lef(x)
        x_enhanced = self.fem(x_low, x)
        y = self.c2f_conv(x_enhanced)
        y1, y2 = y.chunk(2, 1)
        out = [y1, y2]
        for b in self.bottlenecks:
            y2 = b(y2)
            out.append(y2)
        return self.cv2(torch.cat(out, dim=1))


class AMC2fLEFEM(C2fLEFEM):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut)
        self.simam = SimAM()

    def forward(self, x):
        x_low = self.lef(x)
        x_enhanced = self.fem(x_low, x)
        y = self.c2f_conv(x_enhanced)
        y1, y2 = y.chunk(2, 1)
        out = [y1, self.simam(y2)]
        for b in self.bottlenecks:
            y2 = b(y2)
            out.append(y2)
        return self.cv2(torch.cat(out, dim=1))


# ----------------------------------------------------------------------
# 5. Modulo AMRFB
# ----------------------------------------------------------------------
class AMRFB(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c_ = max(1, c1 // 4)
        self.branch1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1),
            nn.Conv2d(c_, c_, 3, padding=1, dilation=1),
            SimAM()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(c1, c_, 1),
            nn.Conv2d(c_, c_, 3, padding=3, dilation=3),
            SimAM()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(c1, c_, 1),
            nn.Conv2d(c_, c_, 3, padding=5, dilation=5),
            SimAM()
        )
        self.shortcut = nn.Conv2d(c1, c_, 1)
        self.conv_out = nn.Sequential(nn.Conv2d(c_ * 4, c2, 1), nn.BatchNorm2d(c2), nn.SiLU())

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        sc = self.shortcut(x)
        out = torch.cat([b1, b2, b3, sc], dim=1)
        return self.conv_out(out)


# ----------------------------------------------------------------------
# 6. Modulo SimSPPF
# ----------------------------------------------------------------------
class SimSPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = nn.Sequential(nn.Conv2d(c1, c_, 1, 1), nn.BatchNorm2d(c_), nn.ReLU(inplace=True))
        self.cv2 = nn.Sequential(nn.Conv2d(c_ * 4, c2, 1, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))