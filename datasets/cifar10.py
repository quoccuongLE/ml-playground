import os

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

cifar_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ])

DSDIR = os.environ.get("DSDIR", "data")
train_dataset = CIFAR10(root=DSDIR, transform=cifar_transforms, train=True, download=True)
test_dataset = CIFAR10(root=DSDIR, transform=cifar_transforms, train=False, download=True)

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
