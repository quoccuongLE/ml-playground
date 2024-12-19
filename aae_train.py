import random
import torch
from torch.optim import Adam

from models.aae import AdversiaralAutoEncoder as AAE
from datasets.mnist import train_loader

from configs.aae_config import (
    x_dim,
    hidden_dim,
    latent_dim,
    lr,
    epochs as num_epochs,
    train_batch_size,
    beta1
)

num_epochs = 1000
seed = random.randint(0, 999)
torch.manual_seed(seed)

ae_weight_path = f"tmp/weights/aae_ae_test_e{num_epochs}_s{seed}.pth"
discriminator_weight_path = f"tmp/weights/aae_discriminator_test_e{num_epochs}_s{seed}.pth"

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
    prior=prior,
    device=device,
    latent_dim=latent_dim,
)
model.to(device=device)

optimizerD = Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerR = Adam(model.autoencoder.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = Adam(model.autoencoder.encoder.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerR = Adam(model.autoencoder.decoder.parameters(), lr=lr, betas=(beta1, 0.999))

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

print("Starting Training Loop...")
print(f"Seed = {seed}")
model.train()

G_losses = []
D_losses = []
R_losses = []

label = torch.full((train_batch_size,), fake_label, dtype=torch.float, device=device)

for epoch in range(num_epochs):
    overall_R_loss, overall_D_loss, overall_G_loss = 0, 0, 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(train_batch_size, x_dim).to(device)

        ############################
        # (1) Reconstruction - Update encoder and decoder
        ###########################
        # model.autoencoder.zero_grad()
        # x_hat, z_mean, log_z_var = model.autoencoder(x, mode=None)
        # errR = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        # errR.backward()
        # optimizerR.step()

        ############################
        # (2a) Regulalization - Update D network: maximize log(D(x)) + log(1 - D(Enc(z)))
        ###########################
        ## Train with all-real latent distribution
        # optimizerD.zero_grad() # The same for the line below
        model.discriminator.zero_grad()
        z_mean, log_z_var = model.autoencoder.encoder(x)
        label.fill_(real_label)
        errD_real = model.discriminator.loss(z_mean, label)
        errD_real.backward()

        ## Train with all-fake batch
        z_prior_samples = model.prior.sample(labels=y)
        label.fill_(fake_label)
        errD_fake = model.discriminator.loss(z_prior_samples, label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2b) Regulalization - Update Generator G Network (a.k.a. Encoder)
        ###########################
        model.autoencoder.encoder.zero_grad()
        # z_mean, log_z_var = model.autoencoder.encoder(x)
        z = model.autoencoder.encoder.reparameterization(
            mean=z_mean, log_var=log_z_var, sample_num=-1
        )
        label.fill_(real_label)
        errG = model.discriminator.loss(z, label)
        optimizerG.step()

        # batch_errR = errR.item() / train_batch_size
        batch_errD = errD.item() / train_batch_size
        batch_errG = errG.item() / train_batch_size
        # overall_R_loss += batch_errR
        overall_D_loss += batch_errD
        overall_G_loss += batch_errG
        # Save Losses for plotting later
        # R_losses.append(batch_errR)
        D_losses.append(batch_errD)
        G_losses.append(batch_errG)

        # Output training stats
    if epoch % 5 == 0:
        mark_string = f"[{epoch:d}/{num_epochs:d}]"
        losses_string = f"\tloss_R: {overall_R_loss / (batch_idx + 1):.4f} | loss_D: {overall_D_loss / (batch_idx + 1):.4f} | loss_G: {overall_G_loss / (batch_idx + 1):.4f}"
        print(f"{mark_string}{losses_string}")


print("Finish!!")
torch.save(model.autoencoder.state_dict(), ae_weight_path)
torch.save(model.discriminator.state_dict(), discriminator_weight_path)
