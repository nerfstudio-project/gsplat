ALPHA_THRESHOLD = 1.0 / 255.0
# MAX_ALPHA and TRANSMITTANCE_THRESHOLD are chosen so that the equivalent of
# a maximal opacity Gaussian has to be rasterized twice to reach the threshold,
# without getting the transmittance too small for numerical stability of
# the backward pass.
# i.e. TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2
MAX_ALPHA = 0.99
TRANSMITTANCE_THRESHOLD = 1e-4

MAX_KERNEL_DENSITY_CUTOFF = 0.0113

# Floor for the antialiased compensation factor (sqrt(det_orig / det_blur)).
# Prevents compensation from reaching zero for extremely small Gaussians.
MIN_COMPENSATION = 0.005
