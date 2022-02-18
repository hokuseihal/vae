import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=(32, 64, 256, 512), **kwargs):
        super(VAE, self).__init__()

        self.dummy_param = nn.Parameter(torch.empty(0))

        modules = []

        # Build Encoder
        prech=in_channels
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
        self.fc_mu = nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 1)
        self.fc_var = nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 1)

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

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=mu.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return self.decoder(z), mu, logvar, z

    def trainbatch(self, x):
        recon_x, mu, logvar, z = self(x)
        bce = F.mse_loss(recon_x, x)
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        genimg = self.generate(size=(z.shape))
        return bce + kld, {
            'stats': {'loss': round((bce + kld).item(), 3), 'mse': round(bce.item(), 3), 'kld': round(kld.item(), 3)},
            'img': {'img': x, 'recon_img': recon_x.detach(), 'genimg': genimg}
        }

    @torch.no_grad()
    def generate(self, z=None, size=None):
        assert not (z is None and size is None)
        if z is None: z = torch.randn(size, device=self.dummy_param.device)
        return self.decoder(z)
