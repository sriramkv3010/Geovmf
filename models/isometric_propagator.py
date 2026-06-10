import torch.nn as nn


class IsometricPropagator(nn.Module):
   
    def __init__(
        self,
        in_dim:  int = 512,
        hid:     int = 512,
        out_dim: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.ReLU(),
            nn.Linear(hid,    hid), nn.LayerNorm(hid), nn.ReLU(),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x):
        return self.net(x)
