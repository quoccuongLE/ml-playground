import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from configs.vae_config import (hidden_dim, latent_dim, weight_path, x_dim)
from datasets.mnist import test_loader
from models.vae import Decoder, Encoder
from models.vae import Model as VAE

batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to("cpu").detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")
        if i > num_batches:
            plt.colorbar()
            break


# Model definition
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = VAE(encoder=encoder, decoder=decoder, device=device)

model.load_state_dict(torch.load(weight_path, weights_only=True))
model.to(torch.device(device))
model.eval()

with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, x_dim)
        x = x.to(device)
        x_hat, _, _ = model(x)
        break

with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(device)
    generated_images = model.decoder(noise)


save_image(generated_images.view(batch_size, 1, 28, 28), "tmp/generated_sample_2.png")
