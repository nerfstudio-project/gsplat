import numpy as np
from scipy.linalg import sqrtm, eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches


def plot_gaussian_3d(ax, mean, cov, color='blue', alpha=0.3, label=None):
    """
    Plots a 3D Gaussian distribution as an ellipsoid.
    
    Parameters:
    - ax: Matplotlib 3D axis
    - mean: 3D mean vector (3,)
    - cov: 3x3 covariance matrix
    - color: Color of the ellipsoid
    - alpha: Transparency
    - label: Legend label
    """
    # Eigen decomposition to get orientation and axes lengths
    eigvals, eigvecs = eigh(cov)  # eigvals: variances, eigvecs: directions

    # Create sphere (parametric representation)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Stack into a 3xN array (flattened coordinates)
    sphere = np.array([x.flatten(), y.flatten(), z.flatten()])

    # Apply eigenvalues (scaling) and eigenvectors (rotation)
    transformed_sphere = eigvecs @ np.diag(np.sqrt(eigvals)) @ sphere

    # Reshape back to 3D form
    x_transformed = transformed_sphere[0, :].reshape(x.shape) + mean[0]
    y_transformed = transformed_sphere[1, :].reshape(y.shape) + mean[1]
    z_transformed = transformed_sphere[2, :].reshape(z.shape) + mean[2]

    # Plot the surface
    ax.plot_surface(x_transformed, y_transformed, z_transformed, color=color, alpha=alpha, linewidth=0.5)

def visualize_gaussians_and_barycenter_3d(means, covariances, bary_mean, bary_cov, weights=None):
    """
    Visualizes a set of 3D Gaussians and their Wasserstein barycenter.

    Parameters:
    - means: List of 3D mean vectors (n_samples, 3)
    - covariances: List of 3x3 covariance matrices (n_samples, 3, 3)
    - bary_mean: 3D mean vector of the barycenter
    - bary_cov: 3x3 covariance matrix of the barycenter
    - weights: List of weights for the Gaussians (optional)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Plot original Gaussians
    for i, (mean, cov) in enumerate(zip(means, covariances)):
        color = colors[i % len(colors)]
        label = f"Gaussian {i+1} (w={weights[i]:.2f})" if weights is not None else f"Gaussian {i+1}"
        plot_gaussian_3d(ax, mean, cov, color=color, alpha=0.3, label=label)
        ax.scatter(*mean, color=color, marker='o', edgecolors='black', s=100, zorder=3)

    # Plot Barycenter
    plot_gaussian_3d(ax, bary_mean, bary_cov, color='black', alpha=0.5, label="Wasserstein Barycenter")
    ax.scatter(*bary_mean, color='black', marker='x', s=150, zorder=4)

    # Labels and view
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("Wasserstein Barycenter of 3D Gaussians")

    plt.legend()
    plt.show()
