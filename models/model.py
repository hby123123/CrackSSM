import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.encoder import CrackEncoder
from .decoders.decoder import Decoder


class Model(nn.Module):
    def __init__(self, num_classes=1, img_size=(448, 448), decoder_depths=(4, 4, 4, 4)):
        super().__init__()
        self.backbone = CrackEncoder()
        encoder_channels = self.backbone.dims
        self.decoder = Decoder(
            img_size=list(img_size),
            in_channels=encoder_channels,
            num_classes=num_classes,
            depths=list(decoder_depths),
            embed_dim=encoder_channels[0],
            deep_supervision=False,
        )

    def forward(self, rgb):
        orisize = rgb.shape
        encoder_features, orientation_map, haar_features = self.backbone(rgb)
        out = self.decoder(encoder_features, haar_features)
        out = F.interpolate(out, size=orisize[2:], mode="bilinear", align_corners=False)
        orientation_map = F.interpolate(
            orientation_map, size=orisize[2:], mode="bilinear", align_corners=False
        )
        return out, orientation_map

    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path, map_location="cpu")
        if isinstance(pretrain_dict, dict) and "model_state_dict" in pretrain_dict:
            pretrain_dict = pretrain_dict["model_state_dict"]
        elif isinstance(pretrain_dict, dict) and "state_dict" in pretrain_dict:
            pretrain_dict = pretrain_dict["state_dict"]
        model_dict = self.state_dict()
        loaded = {k: v for k, v in pretrain_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(loaded)
        self.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(loaded)} layers from {model_path}")


if __name__ == "__main__":
    from thop import profile
    model = Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.randn(1, 3, 448, 448, device=device)
    pred, guide = model(x)
    print("pred_out:", pred.shape, "orientation_map:", guide.shape)
    flops, params = profile(model, inputs=(x,))
    print(f"params: {params / 1e6:.4f} M  flops: {flops / 1e9:.4f} G")
