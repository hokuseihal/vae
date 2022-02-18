import torch
import torch.nn as nn
import torch.nn.functional as F
from model import utils as MU

class MatchingVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=(32, 64, 256, 512), **kwargs):
        super(MatchingVAE, self).__init__()

        self.dummy_param = nn.Parameter(torch.empty(0))

        modules = []

        # Build Encoder
        prech = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prech, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            prech = h_dim

        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = []

        hidden_dims = hidden_dims[::-1]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(
            *modules,
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=in_channels,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

    def trainbatch(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        genimg = self.generate(z.shape)
        reconloss = F.mse_loss(x, recon)
        distloss = MU.distloss(z, torch.randn(z.shape, device=z.device))
        return reconloss + distloss, {
            'stats': {'loss': reconloss.item() + distloss.item(), 'reconloss': reconloss.item(),
                      'distloss': distloss.item()}, 'img': {'img': x, 'recon_img': recon.detach(), 'genimg': genimg}}
    @torch.no_grad()
    def generate(self,size):
        return self.decoder(torch.randn(size,device=self.dummy_param.device))
