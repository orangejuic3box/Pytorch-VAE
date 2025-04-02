import torch
from torch import nn
import torch.nn.functional as F

class MVAE(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size, layers):
        super().__init__()
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        self.layers = layers

        # Encoder
        self.encoder = nn.Sequential(*[
            self._conv(channel_num if i == 0 else kernel_num // (2 ** (self.layers - i)), 
                    kernel_num // (2 ** (self.layers - i - 1)))
            for i in range(self.layers)
        ])


        # # 5 layers
        # self.encoder = nn.Sequential( 
        #     # self._conv(channel_num, kernel_num // 8),  # Smaller filters at first
        #     self._conv(channel_num, kernel_num // 16),
        #     self._conv(kernel_num // 16, kernel_num // 8),
        #     self._conv(kernel_num // 8, kernel_num // 4),
        #     self._conv(kernel_num // 4, kernel_num // 2),
        #     self._conv(kernel_num // 2, kernel_num),  # Increase capacity
        # )
        
        # Encoded feature's size and volume
        self.feature_size = image_size // (2 ** self.layers)#32#16#8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # Latent space aka q?
        self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

        # Projection
        self.project = self._linear(z_size, self.feature_volume, relu=False)

        # Decoder
        self.decoder = nn.Sequential(
            *[
                self._deconv(kernel_num // (2 ** i), kernel_num // (2 ** (i + 1)))
                for i in range(layers - 1)
            ],
            self._deconv(kernel_num // (2 ** (layers - 1)), channel_num),  # Final layer to match input channels
            nn.Sigmoid()
        )

        # # 5 layers
        # self.decoder = nn.Sequential(
        #     self._deconv(kernel_num, kernel_num // 2),
        #     self._deconv(kernel_num // 2, kernel_num // 4),
        #     self._deconv(kernel_num // 4, kernel_num // 8),
        #     self._deconv(kernel_num // 8, kernel_num // 16),
        #     self._deconv(kernel_num // 16, channel_num),
        #     # self._deconv(kernel_num // 8, channel_num),
        #     nn.Sigmoid()
        # )


    



    def apply_xavier_initialization(self):
        # Manually initialize the weights for Conv and Linear layers
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        print("charles xavier was here")

    def forward(self, x):
        # encode x 
        encoded = self.encoder(x)

        # sample latent area z from q given x
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        z_projected = self.project(z).view(
            -1, self.kernel_num, 
            self.feature_size, 
            self.feature_size
        )
        
        # recontruct x from latent space z
        x_reconstructed = self.decoder(z_projected)

        return (mean, logvar), x_reconstructed

    # ==============
    # VAE functions
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        # potentially change this?
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def reconstruction_loss(self, reconstructed_x, x):
        # return F.mse_loss(reconstructed_x, x, reduction='sum')
        return nn.MSELoss(reduction='mean')(reconstructed_x, x)
    
    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()
        # kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'MVAE'
            '-{kernel_num}k'
            '-z{z}'
            '-{label}'
            '-{layers}layers'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            z=self.z_size,
            layers=self.layers
        )

    def sample(self, size):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        z = torch.randn(size, self.z_size, device=device)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected).data
    
    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
    
    # ======
    # Layers
    # ======


    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def _deconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def _linear(self, in_size, out_size, relu=True):
        layers = [nn.Linear(in_size, out_size)]
        if relu:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
