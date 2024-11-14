import numpy as np
import matplotlib.pyplot as plt


def generate_swiss_roll(n_samples, noise=0.0):
    """
    Generates a Swiss roll distribution.

    Args:
        n_samples: The number of samples to generate.
        noise: The amount of noise to add to the distribution.

    Returns:
        A numpy array of shape (n_samples, 3) containing the generated samples.
    """

    t = 3 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = 10 * np.random.rand(1, n_samples)

    X = np.concatenate((x, y, z), axis=0).T
    X += noise * np.random.randn(*X.shape)

    return X


# Generate a Swiss roll with 1000 samples and some noise
n_samples = 1000
noise = 0.1
X = generate_swiss_roll(n_samples, noise)

# Visualize the Swiss roll
fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")s
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t[0], cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
plt.title("Swiss Roll")
plt.savefig("swiss_roll.png")
