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


def get_data_loader(
    batch_size: int,
    mode: str = "train",
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 1,
    pin_memory: bool = True,
):
    return DataLoader(
        dataset=train_dataset if mode == "train" else test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
