import numpy as np
import matplotlib.pyplot as plt


def generate_swiss_roll(n_samples: int, noise: float = 0.0, num_classes: int = 10):
    """
    Generates a Swiss roll distribution.

    Args:
        n_samples: The number of samples to generate.
        noise: The amount of noise to add to the distribution.

    Returns:
        A numpy array of shape (n_samples, 3) containing the generated samples.
    """

    t = num_classes * np.pi / 2 * np.random.rand(1, n_samples)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = 10 * np.random.rand(1, n_samples)

    X = np.concatenate((x, y, z), axis=0).T
    X += noise * np.random.randn(*X.shape)

    return X


class SwissRoll:
    def __init__(
        self,
        num_classes: int = 10,
        beta: float = 0.2,
        L: float = 2.0,
    ):
        k = np.arange(0, num_classes + 1)
        self.beta = beta
        self.L = L
        self.alpha = np.sqrt(2 * L * k / beta)
        self.delta = np.asarray(
            [self.alpha[i + 1] - self.alpha[i] for i in range(num_classes)]
        )

    def generate_arc(self, labels: np.ndarray, noise: float = 0.0):
        t = np.random.rand(len(labels)) * self.delta[labels] + self.alpha[labels]
        x = t * np.cos(t)
        y = t * np.sin(t)
        X = np.stack((x, y)).T
        X += noise * np.random.randn(*X.shape)
        return X


# Generate a Swiss roll with 1000 samples and some noise
n_samples = 1000
noise = 0.5
X = generate_swiss_roll(n_samples, noise)

Y = np.concatenate([x*np.ones(100).astype(int) for x in range(10)])
swiss_roll = SwissRoll()
X = swiss_roll.generate_arc(Y, noise=0.5)

# Visualize the Swiss roll
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=Y, s=10, cmap="tab10", alpha=0.5)
plt.title("Swiss Roll")
plt.savefig("swiss_roll.png")
