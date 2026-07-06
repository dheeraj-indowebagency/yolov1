"""
YOLOv1 network architecture.

Faithfully implements the 24 convolutional layers followed by 2 fully-connected
layers described in Table 1 / Figure 3 of the paper:

  "You Only Look Once: Unified, Real-Time Object Detection"
  Redmon et al., CVPR 2016  --  https://arxiv.org/abs/1506.02640

Layer summary
-------------
Layers  1-20 : convolutional backbone (pretrained on ImageNet in the paper).
Layers 21-24 : detection-specific convolutional layers.
FC-1         : 7*7*1024 -> 4096  (LeakyReLU + Dropout 0.5)
FC-2         : 4096     -> S*S*(B*5+C)   (linear activation)

All hidden layers use Leaky ReLU with slope 0.1.
"""

import torch
import torch.nn as nn

import config


class YOLOv1(nn.Module):
    """YOLOv1 object detector."""

    def __init__(
        self,
        S: int = config.S,
        B: int = config.B,
        C: int = config.C,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        self.features = self._build_conv_layers()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, S * S * (B * 5 + C)),
            # Final layer uses a linear activation (Section 2.2).
        )

        self._initialize_weights()

    # ------------------------------------------------------------------
    # Architecture (Table 1 / Figure 3)
    # ------------------------------------------------------------------
    @staticmethod
    def _conv_block(in_c: int, out_c: int, kernel: int, **kwargs) -> list:
        """Conv2d + BatchNorm + LeakyReLU(0.1).

        BatchNorm is essential for from-scratch training (without ImageNet
        pretrained backbone) to stabilise activations across the deep network.
        Padding keeps spatial dims when stride=1.
        """
        padding = kwargs.pop("padding", kernel // 2)
        return [
            nn.Conv2d(in_c, out_c, kernel, padding=padding, bias=False, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
        ]

    def _build_conv_layers(self) -> nn.Sequential:
        layers: list[nn.Module] = []

        # ---- Block 1 ---- Layer 1  (conv 7x7x64 s2) + MaxPool
        layers += self._conv_block(3, 64, 7, stride=2)
        layers.append(nn.MaxPool2d(2, stride=2))
        # 448 -> 224 -> 112

        # ---- Block 2 ---- Layer 2  (conv 3x3x192) + MaxPool
        layers += self._conv_block(64, 192, 3)
        layers.append(nn.MaxPool2d(2, stride=2))
        # 112 -> 56

        # ---- Block 3 ---- Layers 3-6 + MaxPool
        layers += self._conv_block(192, 128, 1)   # Layer 3
        layers += self._conv_block(128, 256, 3)   # Layer 4
        layers += self._conv_block(256, 256, 1)   # Layer 5
        layers += self._conv_block(256, 512, 3)   # Layer 6
        layers.append(nn.MaxPool2d(2, stride=2))
        # 56 -> 28

        # ---- Block 4 ---- Layers 7-16 + MaxPool
        for _ in range(4):                         # Layers 7-14  (4x [1x1, 3x3])
            layers += self._conv_block(512, 256, 1)
            layers += self._conv_block(256, 512, 3)
        layers += self._conv_block(512, 512, 1)    # Layer 15
        layers += self._conv_block(512, 1024, 3)   # Layer 16
        layers.append(nn.MaxPool2d(2, stride=2))
        # 28 -> 14

        # ---- Block 5a (pretrained) ---- Layers 17-20
        for _ in range(2):                         # Layers 17-20  (2x [1x1, 3x3])
            layers += self._conv_block(1024, 512, 1)
            layers += self._conv_block(512, 1024, 3)
        # 14

        # ---- Block 5b (detection) ---- Layers 21-22
        layers += self._conv_block(1024, 1024, 3)              # Layer 21
        layers += self._conv_block(1024, 1024, 3, stride=2)    # Layer 22
        # 14 -> 7

        # ---- Block 6 (detection) ---- Layers 23-24
        layers += self._conv_block(1024, 1024, 3)              # Layer 23
        layers += self._conv_block(1024, 1024, 3)              # Layer 24
        # 7

        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        """Kaiming (He) init for conv/BN layers; Xavier for FC layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.1
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 3, 448, 448)

        Returns
        -------
        (N, S, S, B*5 + C)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = YOLOv1()
    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = net(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}  (expected [1, {config.S}, {config.S}, "
          f"{config.B * 5 + config.C}])")
    print(f"Params : {count_parameters(net):,}")
