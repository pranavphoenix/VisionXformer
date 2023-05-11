import torch
import torch.fft
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, mlp_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class FourierBlock(nn.Module):
    def __init__(
        self,
        *,
        dim = 16,
        mlp_dim = 32,
        dropout = 0.
    ):
        super().__init__()
        
        self.ff = nn.Sequential(
            nn.BatchNorm2d(dim),
            FeedForward(dim, mlp_dim, dropout = dropout)
        )

        
    def forward(self, x):

        x = torch.fft.fft(torch.fft.fft2(x), dim=1).real
        
        x = self.ff(x)
        
        return x
        
class FNet2D(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        mlp_dim, 
        dropout = 0.,
        
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(FourierBlock(dim = dim, mlp_dim = mlp_dim, dropout = dropout))
        
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim, num_classes)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(3, int(dim/2), 3, 1, 1),
            nn.Conv2d(int(dim/2), dim, 3, 1, 1)
        )

      

    def forward(self, img):
        x = self.conv(img)  
 
        for attn in self.layers:
            x = attn(x) + x

        out = self.pool(x)

        return out
      
