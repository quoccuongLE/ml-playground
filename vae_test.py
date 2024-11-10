import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from models.vae import Encoder, Decoder, Model as VAE
from torchvision.utils import make_grid, save_image
from datasets.mnist import test_loader
from configs.vae_config import (
    x_dim,
    hidden_dim,
    latent_dim,
    lr,
    epochs,
    weight_path,
    batch_size,
)

batch_size = 1
cuda = True
_device = torch.device("cuda" if cuda else "cpu")
# test_loader.batch_size = batch_size

# Model definition
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = VAE(encoder=encoder, decoder=decoder, device=_device)

model.load_state_dict(torch.load(weight_path, weights_only=True))
model.to(torch.device("cuda"))
model.eval()

with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, x_dim)
        x = x.to(_device)
        x_hat, _, _ = model(x)
        break


# def show_image(x, idx):
#     x = x.view(batch_size, 28, 28)

#     fig = plt.figure()
#     plt.imshow(x[idx].cpu().numpy())


with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(_device)
    generated_images = model.decoder(noise)


save_image(generated_images.view(batch_size, 1, 28, 28), "tmp/generated_sample.png")
