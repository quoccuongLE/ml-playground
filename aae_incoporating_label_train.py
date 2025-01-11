import io
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam

from configs.aae_config import epochs as NUM_EPOCHS
from configs.aae_config import (hidden_dim, latent_dim, lr, test_batch_size,
                                train_batch_size, x_dim)
from datasets.mnist import get_data_loader
from models.aae import IncoporatedLabelAAE as AAE
from utils import plot_latent


def save_model(model, ae_weight_path: str, discriminator_weight_path: str):
    torch.save(model.autoencoder.state_dict(), ae_weight_path)
    torch.save(model.discriminator.state_dict(), discriminator_weight_path)


def main(
    prior: str = "gmm",
    num_epochs: int = NUM_EPOCHS,
    seed: int = -1,
    skip_rate_G: int = 1,
    epoch_checkpoint_rate: int = 100,
    save_stats_interval: int = 5,
    device: str = "cuda"
):
    if seed == -1:
        seed = random.randint(0, 999)
    torch.manual_seed(seed)
    if os.environ.get("LAUNCH_MODE") == "debug":
        weight_dir = Path(f"tmp/weights/aae/{prior}/e{num_epochs}/debug")
    else:
        weight_dir = Path(f"tmp/weights/aae/{prior}/e{num_epochs}/s{seed}")
    weight_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=weight_dir
        / f"aae_train_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log",
    )

    ae_weight_path = weight_dir / f"ae_e{num_epochs}.pth"
    discriminator_weight_path = weight_dir / f"discriminator_e{num_epochs}.pth"

    device = torch.device(device)
    num_classes = 10

    encoder = dict(input_dim=x_dim, hidden_dim=hidden_dim, depth=3)
    decoder = dict(output_dim=x_dim, hidden_dim=hidden_dim, depth=3)
    if prior == "gmm":
        prior = dict(
            type="GaussianMultivariateMixture2D",
            num_classes=num_classes,
            radius=2.0,
            sigma_1=2.0,
            sigma_2=0.1,
        )
    elif prior == "swiss_roll":
        prior = dict(
            type="SwissRoll", num_classes=num_classes, beta=0.2, base_length=2.0
        )
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

    optimizerD = Adam(model.discriminator.parameters(), lr=lr)
    optimizerR = Adam(model.autoencoder.parameters(), lr=lr)
    optimizerG = Adam(model.autoencoder.encoder.parameters(), lr=lr)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0
    skip_rate_G = skip_rate_G

    train_loader = get_data_loader(batch_size=train_batch_size, mode="train")
    test_loader = get_data_loader(batch_size=test_batch_size, mode="test")
    print("Starting Training Loop...")
    print(f"Seed = {seed}")
    model.train()

    G_losses = []
    D_losses = []
    R_losses = []

    label = torch.full(
        (train_batch_size,), fake_label, dtype=torch.float, device=device
    )

    for epoch in range(num_epochs):
        overall_R_loss, overall_D_loss, overall_G_loss = 0, 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view(train_batch_size, x_dim).to(device)

            ############################
            # (1) Reconstruction - Update encoder and decoder
            ###########################
            model.autoencoder.zero_grad()
            x_hat, z_mean, _ = model.autoencoder(x, mode=None)
            errR = F.binary_cross_entropy(x_hat, x, reduction="sum")
            errR.backward()
            optimizerR.step()

            ############################
            # (2a) Regulalization - Update D network: maximize log(D(x)) + log(1 - D(Enc(z)))
            ###########################
            ## Train with all-generated latent distribution
            # optimizerD.zero_grad() # The same for the line below
            model.discriminator.zero_grad()
            with torch.no_grad():
                z_mean, _ = model.autoencoder.encoder(x)
            label.fill_(fake_label)
            errD_fake = model.discriminator_loss(z_mean, y, label)
            errD_fake.backward()

            ## Train with all-true prior latent distribution
            z_prior_samples = model.prior.sample(labels=y).squeeze()
            label.fill_(real_label)
            errD_real = model.discriminator_loss(z_prior_samples, y, label)
            errD_real.backward()
            errD = errD_real + errD_fake

            # model.discriminator.zero_grad()
            # with torch.no_grad():
            #     z_mean, _ = model.autoencoder.encoder(x)
            # z_prior_samples = model.prior.sample(labels=y).squeeze()
            # real_labels = torch.ones(train_batch_size, device=device)
            # fake_labels = torch.zeros(train_batch_size, device=device)
            # z = torch.cat((z_mean, z_prior_samples), dim=0)
            # labels = torch.cat((fake_labels, real_labels), dim=0)
            # errD = model.discriminator_loss(x=z, y=torch.cat((y, y)), labels=labels)
            # errD.backward()
            optimizerD.step()
            batch_errD = errD.item() / train_batch_size / 2

            if batch_idx % skip_rate_G == 0:
                ############################
                # (2b) Regulalization - Update Generator G Network (a.k.a. Encoder)
                ###########################
                model.autoencoder.encoder.zero_grad()
                label.fill_(real_label)
                z_mean, _ = model.autoencoder.encoder(x)
                errG = model.discriminator_loss(x=z_mean, y=y, labels=label)
                errG.backward()
                optimizerG.step()
                batch_errG = errG.item() / train_batch_size

            batch_errR = errR.item() / train_batch_size
            overall_R_loss += batch_errR
            overall_D_loss += batch_errD
            overall_G_loss += batch_errG
            # Save Losses for plotting later
            R_losses.append(batch_errR)
            D_losses.append(batch_errD)
            G_losses.append(batch_errG)

        # Output training stats
        if epoch % save_stats_interval == 0:
            mark_string = f"[{epoch:d}/{num_epochs:d}]"
            losses_string = f"\tloss_R: {overall_R_loss / (batch_idx + 1):.4f} |"
            losses_string += f" loss_D: {overall_D_loss / (batch_idx + 1):.4f} |"
            losses_string += f" loss_G: {overall_G_loss / (batch_idx + 1):.4f}"
            print(f"{mark_string}{losses_string}")
            logging.info(f"{mark_string}{losses_string}")
            plot_latent(
                autoencoder=model.autoencoder,
                test_batch_size=test_batch_size,
                data_loader=test_loader,
                x_dim=x_dim,
                save_img_path=weight_dir / f"latent_e{epoch}.png",
            )
            shutil.copyfile(
                weight_dir / f"latent_e{epoch}.png", weight_dir / f"latent_latest.png"
            )

        if epoch % epoch_checkpoint_rate == 0:
            save_model(
                model=model,
                ae_weight_path=weight_dir / f"ae_e{epoch}.pth",
                discriminator_weight_path=weight_dir / f"discriminator_e{epoch}.pth",
            )

    print("Finish!!")
    logging.info("Finish!!")
    logging.info(f"Seed = {seed}")

    torch.save(model.autoencoder.state_dict(), ae_weight_path)
    torch.save(model.discriminator.state_dict(), discriminator_weight_path)


if __name__ == "__main__":
    fire.Fire(main)
