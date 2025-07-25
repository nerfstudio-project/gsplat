import numpy as np
from scipy.linalg import sqrtm, eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches

def wasserstein_barycenter_gaussians_orig(means, covariances, weights=None, max_iter=100, tol=1e-6):
    """
    Compute the Wasserstein-2 barycenter of a set of Gaussian distributions.
    
    Parameters:
    - means: list of 2D mean vectors (n_samples, 2)
    - covariances: list of 2x2 covariance matrices (n_samples, 2, 2)
    - weights: np.array of weights (n_samples,). If None, assumes uniform weights.
    - max_iter: maximum number of iterations for fixed-point algorithm.
    - tol: tolerance for convergence.

    Returns:
    - bary_mean: 2D mean vector of the barycenter.
    - bary_cov: 2x2 covariance matrix of the barycenter.
    """
    n = len(means)
    dim = len(means[0])  # Should be 3 for 3D Gaussians

    if weights is None:
        weights = np.ones(n) / n  # Equal weights by default

    # Compute mean barycenter (weighted average of means)
    bary_mean = np.sum(weights[:, None] * means, axis=0)

    # Initialize barycenter covariance as the weighted sum
    weights /= weights.sum() 
    bary_cov = np.sum(weights[i] * covariances[i] for i in range(n))
    
    for _ in range(max_iter):
        bary_cov_prev = bary_cov.copy()
        sqrt_bary_cov = sqrtm(bary_cov)

        # Compute new covariance using the fixed-point update
        bary_cov = np.zeros((dim, dim))
        for i in range(n):
            sqrt_term = sqrtm(sqrt_bary_cov @ covariances[i] @ sqrt_bary_cov)
            bary_cov += weights[i] * sqrt_term

        #bary_cov = bary_cov @ bary_cov  # Square the result

        # Convergence check
        if np.linalg.norm(bary_cov - bary_cov_prev, ord='fro') < tol:
            break

    return bary_mean, bary_cov

def wasserstein_barycenter_gaussians(means, covariances, weights=None, max_iter=100, tol=1e-6):
    """
    I use this function for testing different variants of barycenter computation. Use wasserstein_barycenter_gaussians_orig for the original implementation.
    Compute the Wasserstein-2 barycenter of a set of Gaussian distributions.
    
    Parameters:
    - means: list of 2D mean vectors (n_samples, 2)
    - covariances: list of 2x2 covariance matrices (n_samples, 2, 2)
    - weights: np.array of weights (n_samples,). If None, assumes uniform weights.
    - max_iter: maximum number of iterations for fixed-point algorithm.
    - tol: tolerance for convergence.

    Returns:
    - bary_mean: 2D mean vector of the barycenter.
    - bary_cov: 2x2 covariance matrix of the barycenter.
    """
    # print(f"means: {means}")
    # print(f"covariances: {covariances}")
    # print(f"weights: {weights}")
    
    assert len(means) == len(covariances), "Number of means and covariances must match"
    assert len(means) > 0, "No means provided"
    
    n = len(means)
    if n == 1:
        return means[0], covariances[0]
    
    dim = len(means[0])  # Should be 3 for 3D Gaussians
    
    assert len(means[0]) == 3
    assert len(covariances[0]) == 3
    assert len(covariances[0][0]) == 3

    if weights is None:
        weights = np.ones(n) / n  # Equal weights by default

    # Compute mean barycenter (weighted average of means)
    bary_mean = np.sum(weights[:, None] * means, axis=0)

    # Initialize barycenter covariance as the weighted sum
    weights /= weights.sum() 
    bary_cov = np.sum(weights[i] * covariances[i] for i in range(n))
    # print("bary_cov dtype: ", bary_cov.dtype)
    
    # Check if the covariance matrix is positive definite
    # for i in range(n):
    #     print(f"Eigenvalues: {np.linalg.eigvals(covariances[i])}")
    #     if not np.all(np.linalg.eigvals(covariances[i]) > 0):
    #         print(f"Covariance matrix {i} is not positive definite")
    #         print(f"Covariance matrix {i}: {covariances[i]}")
    #         print(f"Eigenvalues: {np.linalg.eigvals(covariances[i])}")
    #         print(f"Eigenvectors: {np.linalg.eig(covariances[i])}")
    #         print(f"Eigenvectors: {np.linalg.eig(covariances[i])}")
    for _ in range(max_iter):
        bary_cov_prev = bary_cov.copy()
        sqrt_bary_cov = sqrtm(bary_cov)
        # print(f"sqrt_bary_cov dtype: {sqrt_bary_cov.dtype}")

        # Compute new covariance using the fixed-point update
        bary_cov = np.zeros((dim, dim))
        for i in range(n):
            # print(f"Covariance matrix {i}: {covariances[i]}")
            # print(f"sqrt_bary_cov: {sqrt_bary_cov}")
            # print(f"covariances[i] dtype: {covariances[i].dtype}")
            sqrt_term = sqrtm(sqrt_bary_cov @ covariances[i] @ sqrt_bary_cov.T)
            # print(f"sqrt_term dtype: {sqrt_term.dtype}")
            # print(f"sqrt_term: {sqrt_term}")
            bary_cov += weights[i] * sqrt_term
        # print(f"bary_cov dtype: {bary_cov.dtype}")

        #bary_cov = bary_cov @ bary_cov  # Square the result

        # Convergence check
        if np.linalg.norm(bary_cov - bary_cov_prev, ord='fro') < tol:
            break

    return bary_mean, bary_cov

def get_w2_distance(means, covariances, bary_mean, bary_cov):
    distances = []
    for i, (mean, cov) in enumerate(zip(means, covariances)):
        mean_diff = np.linalg.norm(mean - bary_mean)**2
        # Square root of the covariance matrix
        sqrt_bary_sigma = sqrtm(bary_cov)
    
        # Wasserstein-2 distance calculation
        cov_sqrt_term = sqrtm(sqrt_bary_sigma @ cov @ sqrt_bary_sigma)
        cov_diff = np.trace(cov + bary_cov - 2 * cov_sqrt_term)
        distance =  np.sqrt(mean_diff + cov_diff)
        distances.append(distance)
    
    mean_w2_distance = np.mean(distances)
    return mean_w2_distance


if __name__ == "__main__":
    # # Example usage 1:
    # means = [np.array([10,2, 3]), np.array([3, 1, 2]), np.array([2, 3, 1])]
    # covariances = [
    #     np.array([[1, 0.3, 0.2], [0.3, 1.5, 0.4], [0.2, 0.4, 1]]),
    #     np.array([[1.2, -0.2, 0.1], [-0.2, 1, 0.3], [0.1, 0.3, 1.3]]),
    #     np.array([[1, 0.1, -0.3], [0.1, 1.2, 0.2], [-0.3, 0.2, 1.1]])
    # ]
    # weights = np.array([0.3, 0.4, 0.3])  # Optional weighting


    # bary_mean, bary_cov = wasserstein_barycenter_gaussians(means, covariances, weights)
    # print("Barycenter Mean:", bary_mean)
    # print("Barycenter Covariance:\n", bary_cov)

    # mean_w2_distance = get_w2_distance(means, covariances, bary_mean, bary_cov)
    # print("Mean W2-distance:", mean_w2_distance)
    
    # Example usage 2:
    
    means = [
            np.array([-0.137571, -0.943233,  0.044001], dtype=np.float64), 
            np.array([-0.135571, -0.943233,  0.043001], dtype=np.float64)
        ]
    covariances = [
            np.array([
                        [ 1.3279008e-04, -2.2325564e-04,  1.6518707e-05],
                        [-2.2325564e-04,  3.7692496e-04, -2.7819591e-05],
                        [ 1.6518707e-05, -2.7819589e-05,  2.0610182e-06]], dtype=np.float64),
            np.array([
                        [ 6.1951432e-05, -2.0695481e-04, -6.6253946e-05],
                        [-2.0695479e-04,  6.9140189e-04,  2.2134058e-04],
                        [-6.6253946e-05,  2.2134060e-04,  7.0861868e-05]], dtype=np.float64)
        ]
    weights = np.array([0.42149818, 0.5785018], dtype=np.float64)  # Optional weighting
    
    bary_mean, bary_cov = wasserstein_barycenter_gaussians(means, covariances, weights)
    print("Barycenter Mean:", bary_mean)
    print("Barycenter Covariance:\n", bary_cov)

    mean_w2_distance = get_w2_distance(means, covariances, bary_mean, bary_cov)
    print("Mean W2-distance:", mean_w2_distance)
    
    