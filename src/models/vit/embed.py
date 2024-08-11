from torch import nn
from timm import create_model
from torchvision.transforms import Normalize

class SwinModel(nn.Module):
    def __init__(self, pretrained_model="swinv2-cr-t-224", device="cuda") -> None:
        """
        vit_tiny_patch16_224.augreg_in21k_ft_in1k
        swinv2_cr_tiny_ns_224.sw_in1k
        """
        super().__init__()
        self.device = device
        self.pretrained_model = pretrained_model
        if pretrained_model == "swinv2-cr-t-224":
            self.pretrained = create_model(
                "swinv2_cr_tiny_ns_224.sw_in1k",
                pretrained=True,
                features_only=True,
                out_indices=[-4, -3, -2, -1],
            ).to(device)
        elif pretrained_model == "swinv2-t-256":
            self.pretrained = create_model(
                "swinv2_tiny_window16_256.ms_in1k",
                pretrained=True,
                features_only=True,
                out_indices=[-4, -3, -2, -1],
            ).to(device)
        elif pretrained_model == "swinv2-cr-s-224":
            self.pretrained = create_model(
                "swinv2_cr_small_ns_224.sw_in1k",
                pretrained=True,
                features_only=True,
                out_indices=[-4, -3, -2, -1],
            ).to(device)
        else:
            raise NotImplementedError

        self.pretrained.eval()
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.upsample = nn.Upsample(scale_factor=2)

        for params in self.pretrained.parameters():
            params.requires_grad = False

    def forward(self, x):
        outputs = self.pretrained(x)
        if self.pretrained_model in ["swinv2-t-256"]:
            for i in range(len(outputs)):
                outputs[i] = outputs[i].permute(0, 3, 1, 2) # Change channel-last to channel-first
        outputs = [self.upsample(feat) for feat in outputs]

        return outputs