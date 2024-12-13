import torch
import torch.nn as nn
from torch.optim import Adam

from models.aae import AdversiaralAutoEncoder as AAE
from datasets.mnist import train_loader

from configs.aae_config import (
    x_dim,
    hidden_dim,
    latent_dim,
    lr,
    epochs as num_epochs,
    weight_path,
    train_batch_size,
    beta1
)

cuda = True
device = torch.device("cuda" if cuda else "cpu")

encoder = dict(input_dim=x_dim, hidden_dim=hidden_dim, depth=3)
decoder = dict(output_dim=x_dim, hidden_dim=hidden_dim, depth=3)
prior = dict(type="GaussianMultivariateMixture2D", num_classes=10, radius=4.0, sigma_1=2.0, sigma_2=0.1)
autoencoder = dict(encoder=encoder, decoder=decoder)
discriminator = dict(hidden_dim=hidden_dim, depth=3)
model: AAE = AAE(
    autoencoder=autoencoder,
    discriminator=discriminator,
    device=device,
    latent_dim=latent_dim,
).to(device)

optimizerD = Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = Adam(model.autoencoder.parameters(), lr=lr, betas=(beta1, 0.999))

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

print("Starting Training Loop...")
model.train()

for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real latent distribution
        # optimizerD.zero_grad() # The same for the line below
        model.discriminator.zero_grad()
        x = x.view(train_batch_size, x_dim).to(device)
        x_hat, z_mean, log_z_var = model.autoencoder(x, mode=None)
        label = torch.full(
            (train_batch_size,), real_label, dtype=torch.float, device=device
        )
        errD_real = model.discriminator.loss(z_mean, label)
        errD_real.backward()

        ## Train with all-fake batch
        z_prior_samples = model.prior.sample(batch_size=train_batch_size)
        label.fill_(fake_label)
        errD_fake = model.discriminator.loss(z_prior_samples, label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # TODO:
        model.autoencoder.zero_grad()
        label.fill_(real_label)
        errG = model.autoencoder.loss()
        # RE = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
