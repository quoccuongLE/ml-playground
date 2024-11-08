import matplotlib.pyplot as plt


test_dataset = MNIST(
    dataset_path, transform=mnist_transform, train=False, download=True
)

test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs
)
