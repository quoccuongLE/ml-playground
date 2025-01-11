import numpy as np
import matplotlib.pyplot as plt


def generate_circular_gaussian_mixture(num_gaussians, num_samples_per_gaussian, radius):
    """
    Generates a mixture of 2D Gaussians distributed on a circle.

    Args:
        num_gaussians: The number of Gaussians in the mixture.
        num_samples_per_gaussian: The number of samples to generate from each Gaussian.
        radius: The radius of the circle on which the Gaussians are distributed.

    Returns:
        A numpy array of shape (num_gaussians * num_samples_per_gaussian, 2) containing the generated samples.
    """
    def cov_rotation(cov, angle_rad):
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        return rotation_matrix @ cov @ rotation_matrix.T

    data = []
    labels = []
    angles = np.linspace(0, 2 * np.pi, num_gaussians, endpoint=False)
    for i, angle in enumerate(angles):
        mean = radius * np.array([np.cos(angle), np.sin(angle)])
        cov = np.array([[0.5, 0], [0, 0.1]])  # Adjust covariance as needed
        samples = np.random.multivariate_normal(
            mean, cov_rotation(cov, angle), num_samples_per_gaussian
        )
        data.append(samples)
        labels.extend([i for _ in range(num_samples_per_gaussian)])

    data = np.vstack(data)
    return data, labels


# Generate 10 Gaussians with 100 samples each on a circle of radius 5
num_gaussians = 10
num_samples_per_gaussian = 500
radius = 4
data, labels = generate_circular_gaussian_mixture(
    num_gaussians, num_samples_per_gaussian, radius
)

# Visualize the generated data
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=1)
plt.title("Mixture of 10 2D Gaussians on a Circle")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")  # Ensure equal aspect ratio for circular visualization
plt.colorbar()
plt.savefig("gmm2.png")
