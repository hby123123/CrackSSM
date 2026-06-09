import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

from .vmamba import Backbone_VSSM_tiny, DPSM


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, x):
        b, _, h, w = x.size()
        xx = torch.arange(w, device=x.device).float() / max(w - 1, 1) * 2 - 1
        yy = torch.arange(h, device=x.device).float() / max(h - 1, 1) * 2 - 1
        xx = xx.view(1, 1, 1, w).expand(b, 1, h, w)
        yy = yy.view(1, 1, h, 1).expand(b, 1, h, w)
        out = torch.cat([x, xx, yy], dim=1)
        if self.with_r:
            rr = torch.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
            out = torch.cat([out, rr], dim=1)
        return out


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
        self.addcoords = AddCoords()

    def forward(self, x):
        return self.conv(self.addcoords(x))


class DirectionalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels=4, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            CoordConv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )

    def forward(self, x):
        return self.conv(x)


class CrackEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.vssm_r = Backbone_VSSM_tiny(**kwargs)
        dims = self.vssm_r.dims
        self.pred_saliency = DirectionalFeatureExtractor(in_channels=dims[-1], out_channels=4)
        self.direction_mamba = nn.ModuleList(
            DPSM(d_model=dims[i], mlp_ratio=0.0, d_state=4, flip=False)
            for i in range(len(dims) - 1)
        )
        self.dwt = DWTForward(J=4, wave="haar", mode="zero")

    @property
    def dims(self):
        return self.vssm_r.dims

    def forward(self, x):
        B = x.shape[0]
        _, x_haar = self.dwt(x)
        haar_features = {"h2": x_haar[0], "h4": x_haar[1], "h8": x_haar[2], "h16": x_haar[3]}
        feats = self.vssm_r(x)
        deep = F.interpolate(feats[3], scale_factor=8, mode="bilinear", align_corners=False)
        orientation_map = self.pred_saliency(deep)
        with torch.no_grad():
            guide = torch.sigmoid(orientation_map)
            guide_up = F.interpolate(guide, scale_factor=8, mode="bilinear", align_corners=False)
        outs = []
        for i in range(4):
            t = feats[i].permute(0, 2, 3, 1).contiguous()
            if i < 3:
                h, w = t.shape[1], t.shape[2]
                g = F.interpolate(guide_up, size=(h, w), mode="bilinear", align_corners=False)
                q = torch.quantile(g.view(B, 4, -1), 0.8, dim=2, keepdim=True).unsqueeze(3)
                g = (g >= q).float()
                t = self.direction_mamba[i](t, g)
            t = t.permute(0, 3, 1, 2).contiguous()
            outs.append(t)
        return outs, orientation_map, haar_features
