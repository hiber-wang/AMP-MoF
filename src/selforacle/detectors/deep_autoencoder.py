import torch
import torch.nn as nn
import sys
IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH = 3, 224, 224
INPUT_SHAPE = (IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH, )
from .anomaly_detector import AnomalyDetector

class Model(nn.Module):
    def __init__(self, hidden_layer=256):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(IMAGE_CHANNELS  * IMAGE_HEIGHT * IMAGE_WIDTH, hidden_layer),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer//2, hidden_layer//4),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
           nn.Linear(hidden_layer//4, hidden_layer//2),
           nn.LeakyReLU(),
           nn.Linear(hidden_layer//2, hidden_layer),
           nn.LeakyReLU(),
           nn.Linear(hidden_layer, IMAGE_CHANNELS  * IMAGE_HEIGHT * IMAGE_WIDTH),
           nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(self.encoder(x))
        return x


class VAE(nn.Module):

    def __init__(self, latent_size=2, y_size=0):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        self.encoder_forward = nn.Sequential(
            nn.Linear(IMAGE_CHANNELS  * IMAGE_HEIGHT * IMAGE_WIDTH + y_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2 * latent_size),
            nn.Sigmoid()
        )

        self.decoder_forward = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, IMAGE_CHANNELS  * IMAGE_HEIGHT * IMAGE_WIDTH),
            nn.Sigmoid()
        )

    def encoder(self, X):
        out = self.encoder_forward(X)
        mu = out[:, :self.latent_size]
        log_var = out[:, self.latent_size:]
        return mu, log_var

    def decoder(self, z):
        mu_prime = self.decoder_forward(z)
        return mu_prime

    def reparameterization(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        z = mu + epsilon * torch.sqrt(log_var.exp())
        return z

    def loss(self, X, mu_prime, mu, log_var):
        # reconstruction_loss = F.mse_loss(mu_prime, X, reduction='mean') is wrong!
        reconstruction_loss = torch.mean(torch.square(X - mu_prime).sum(dim=1))
        latent_loss = torch.mean(0.5 * (log_var.exp() + torch.square(mu) - log_var-1).sum(dim=1))
        return reconstruction_loss + latent_loss

    def forward(self, X, *args, **kwargs):
        mu, log_var = self.encoder(X)
        z = self.reparameterization(mu, log_var)
        mu_prime = self.decoder(z)
        return mu_prime, mu, log_var


class DeepAutoencoder(AnomalyDetector):
    def __init__(self, name, args, hidden_layer_dim=1024, latent_size=128):
        super(DeepAutoencoder, self).__init__(name=name, args=args)
        self.hidden_layer_dim = hidden_layer_dim
        self.latent_size =latent_size

    def get_input_shape(self):
        return INPUT_SHAPE

    def _create_model(self):
        if self.name == "DAE":
            model = Model(hidden_layer=self.hidden_layer_dim)
        elif self.name == "VAE":
            model = VAE(latent_size=self.latent_size)
        return model