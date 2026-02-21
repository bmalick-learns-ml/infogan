import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, init_channels: int = 1024, latent_dim: int = 62, img_channels: int = 1, num_classes: int = 10, num_codes: int = 2):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.num_codes = num_codes
        
        self.lin_z = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.lin_c1 = nn.Linear(in_features=num_classes, out_features=num_classes)
        self.lin_c2 = nn.Linear(in_features=num_codes, out_features=num_codes)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=latent_dim+num_classes+num_codes, out_features=init_channels),
            nn.ReLU(), nn.BatchNorm1d(init_channels)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=init_channels, out_features=128*7*7),
            nn.ReLU(), nn.BatchNorm1d(128*7*7)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64), # output shape: (64, 14, 14)
            nn.ConvTranspose2d(in_channels=64, out_channels=img_channels, stride=2, kernel_size=4, padding=1),
            nn.Tanh() # output shape: (64, 28, 28)
        )

    def forward(self, z, c1, c2):
        lz  = F.relu(self.lin_z(z))
        lc1 = F.relu(self.lin_c1(c1))
        lc2 = F.relu(self.lin_c2(c2))
        out = torch.cat((lz,lc1,lc2), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.view(-1, 128, 7, 7)
        out = self.conv(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, init_channels: int = 64, img_channels: int = 1, num_classes:int = 10, num_codes: int = 2, alpha: float = 0.1):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=init_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(alpha), # output shape: (init_channels, 14, 14)
            nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels*2),
            nn.LeakyReLU(alpha) # output shape: (init_channels*2, 7, 7)
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(in_features=init_channels*2*7*7, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(alpha),
        )
        self.head = nn.Linear(in_features=1024, out_features=1)
        self.q_net = nn.Sequential(
            nn.Linear(in_features=1024, out_features=128),
            nn.BatchNorm1d(128), 
            nn.LeakyReLU(alpha),
        )
        self.q_c1 = nn.Linear(128, num_classes)
        self.q_c2 = nn.Linear(128, num_codes)
      
        
    def forward(self, x):
        out = self.conv(x)
        features = out.view(x.size(0), -1)
        shared = self.fc_shared(features)
        d_out = self.head(shared)  # real or fake
        q = self.q_net(shared)
        c1 = self.q_c1(q)
        c2 = self.q_c2(q)
        return d_out, c1, c2
