import torch
import torch.nn as nn
from torch.optim import Adam

from models.aae import AdversiaralAutoEncoder as AAE
from datasets.mnist import test_loader

from configs.aae_config import x_dim, hidden_dim, latent_dim, test_batch_size

from utils import plot_latent


num_epochs = 95
ae_weight_path = f"tmp/weights/aae_ae_e{num_epochs}.pth"
discriminator_weight_path = f"tmp/weights/aae_discriminator_e{num_epochs}.pth"

cuda = True
device = torch.device("cuda" if cuda else "cpu")

encoder = dict(input_dim=x_dim, hidden_dim=hidden_dim, depth=3)
decoder = dict(output_dim=x_dim, hidden_dim=hidden_dim, depth=3)
prior = dict(
    type="GaussianMultivariateMixture2D",
    num_classes=10,
    radius=4.0,
    sigma_1=2.0,
    sigma_2=0.1,
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

model.autoencoder.load_state_dict(torch.load(ae_weight_path, weights_only=True))
model.discriminator.load_state_dict(
    torch.load(discriminator_weight_path, weights_only=True)
)
model.to(device=device)
model.eval()

plot_latent(
    autoencoder=model.autoencoder,
    test_batch_size=test_batch_size,
    data_loader=test_loader,
    x_dim=x_dim,
    save_img_path="latent_aae_gmm2d_e120.png",
)
