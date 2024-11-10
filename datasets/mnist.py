import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from configs.vae_config import train_batch_size, test_batch_size, dataset_path


mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

kwargs = {"num_workers": 1, "pin_memory": True}

train_dataset = MNIST(
    dataset_path, transform=mnist_transform, train=True, download=True
)
test_dataset = MNIST(
    dataset_path, transform=mnist_transform, train=False, download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    drop_last=True,
    **kwargs
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    shuffle=True,
    drop_last=True,
    **kwargs
)
