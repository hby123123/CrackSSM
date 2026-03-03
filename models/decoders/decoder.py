import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class SimpleUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, x_skip):
        x = self.upsample(x)
        if x.shape[2:] != x_skip.shape[2:]:
            x = F.interpolate(x, size=x_skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x_skip], dim=1)
        return self.conv(x)


class FinalUpsample_X4(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.linear1(x).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous()
        x = self.linear2(x).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x



class WGM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 4):
        super().__init__()
        hidden_channels = max(channels // reduction_ratio, 1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, feature_map: torch.Tensor, single_haar_high_freq: torch.Tensor) -> torch.Tensor:
        high_freq_energy = torch.mean(torch.abs(single_haar_high_freq), dim=1, keepdim=True)
        gate = self.gate_conv(high_freq_energy)
        return feature_map * gate


class GRO(nn.Module):
    def __init__(self, pred_channels: int, haar_channels: int, mid_channels: int = 16,
                 norm_layer: nn.Module = nn.BatchNorm2d):
        super().__init__()
        self.pred_transform = nn.Sequential(nn.Conv2d(pred_channels, mid_channels, kernel_size=1, bias=False),
                                            norm_layer(mid_channels), nn.ReLU(inplace=True))
        self.haar_transform = nn.Sequential(nn.Conv2d(haar_channels, mid_channels, kernel_size=1, bias=False),
                                            norm_layer(mid_channels), nn.ReLU(inplace=True))
        self.similarity_generator = nn.Conv2d(mid_channels, pred_channels, kernel_size=1)
        if pred_channels != haar_channels:
            self.haar_aligner = nn.Conv2d(haar_channels, pred_channels, kernel_size=1)
        else:
            self.haar_aligner = nn.Identity()

    def forward(self, pred_feat: torch.Tensor, haar_feat: torch.Tensor) -> torch.Tensor:
        if pred_feat.shape[2:] != haar_feat.shape[2:]: raise ValueError(
            f"Spatial dimensions of prediction ({pred_feat.shape[2:]}) and Haar features ({haar_feat.shape[2:]}) must match.")
        pred_key = self.pred_transform(pred_feat)
        haar_query = self.haar_transform(haar_feat)
        interaction = pred_key * haar_query
        similarity_map = torch.sigmoid(self.similarity_generator(interaction))
        aligned_haar = self.haar_aligner(haar_feat)
        refined_feat = (1 - similarity_map) * pred_feat + similarity_map * aligned_haar
        return refined_feat


class Decoder(nn.Module):
    def __init__(self,
                 img_size=[448, 448],
                 in_channels=[16, 32, 64, 128],
                 num_classes=1,
                 embed_dim=16,
                 patch_size=4,
                 depths=[4, 4, 4, 4],
                 mlp_ratio=4.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 deep_supervision=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.deep_supervision = deep_supervision
        self.refiner = GRO(pred_channels=embed_dim, haar_channels=3, mid_channels=16)

        num_gating_stages = len(depths) - 1
        self.haar_gates = nn.ModuleList()
        for i in range(num_gating_stages):
            current_channels = in_channels[self.num_layers - 2 - i]
            directional_gates = nn.ModuleList([
                WGM(channels=current_channels),
                WGM(channels=current_channels),
                WGM(channels=current_channels)
            ])
            self.haar_gates.append(directional_gates)

        self.fusion_convs = nn.ModuleList()
        for i in range(num_gating_stages):
            current_channels = in_channels[self.num_layers - 2 - i]
            self.fusion_convs.append(
                nn.Conv2d(current_channels * 4, current_channels, kernel_size=1)
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            curr_in = in_channels[-1 - i]
            skip_in = in_channels[-2 - i]
            out_c = skip_in
            self.layers_up.append(SimpleUpBlock(curr_in, skip_in, out_c))

        self.norm_up = norm_layer(embed_dim)

        if self.deep_supervision:
            self.norm_ds = nn.ModuleList(
                [norm_layer(in_channels[2 - i_layer]) for i_layer in range(self.num_layers - 1)])
            self.output_ds = nn.ModuleList([nn.Conv2d(in_channels=in_channels[2 - i_layer],
                                                      out_channels=self.num_classes, kernel_size=1, bias=False) for
                                            i_layer in range(self.num_layers - 1)])

        self.up = FinalUpsample_X4(input_resolution=(img_size[0] // patch_size, img_size[1] // patch_size),
                                   patch_size=4, dim=embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

    def forward_up_features(self, inputs, haar_features_dict):
        if not self.deep_supervision:
            haar_keys = ['h16', 'h8', 'h4']
            y = None

            for inx, layer_up in enumerate(self.layers_up):
                if inx == 0:
                    x1, x2 = inputs[3 - inx], inputs[3 - inx - 1]
                    y = layer_up(x1, x2)
                elif inx < 3:
                    x1 = y + inputs[3 - inx]
                    x2 = inputs[3 - inx - 1]
                    y = layer_up(x1, x2)
                else:
                    break
                if inx < len(self.haar_gates):
                    y_for_gating = y
                    haar_5d = haar_features_dict[haar_keys[inx]]
                    haar_lh = haar_5d[:, :, 0, :, :]
                    haar_hl = haar_5d[:, :, 1, :, :]
                    haar_hh = haar_5d[:, :, 2, :, :]
                    gated_lh = self.haar_gates[inx][0](y_for_gating, haar_lh)
                    gated_hl = self.haar_gates[inx][1](y_for_gating, haar_hl)
                    gated_hh = self.haar_gates[inx][2](y_for_gating, haar_hh)
                    concatenated_features = torch.cat([y_for_gating, gated_lh, gated_hl, gated_hh], dim=1)
                    y = self.fusion_convs[inx](concatenated_features)

            y = y.permute(0, 2, 3, 1)
            x = self.norm_up(y)
            x_for_refine = x.permute(0, 3, 1, 2).contiguous()

            matching_haar_5d = haar_features_dict['h4']
            matching_haar_4d = torch.mean(matching_haar_5d, dim=1)

            refined_x_chw = self.refiner(x_for_refine, matching_haar_4d)
            x_final_refined = refined_x_chw.permute(0, 2, 3, 1).contiguous()

            return x_final_refined

        else:
            pass

    def forward(self, inputs, haar_features_dict):
        if not self.deep_supervision:
            x = self.forward_up_features(inputs, haar_features_dict)
            x_last = self.up_x4(x, self.patch_size)
            return x_last
        else:
            pass
    def up_x4(self, x, pz):
        B, H, W, C = x.shape
        x = self.up(x)
        x = x.view(B, pz * H, pz * W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.output(x)
        return x
