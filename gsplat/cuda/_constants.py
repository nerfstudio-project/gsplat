ALPHA_THRESHOLD = 1.0 / 255.0
# MAX_ALPHA and TRANSMITTANCE_THRESHOLD are chosen so that the equivalent of
# a maximal opacity Gaussian has to be rasterized twice to reach the threshold,
# without getting the transmittance too small for numerical stability of
# the backward pass.
# i.e. TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2
MAX_ALPHA = 0.99
TRANSMITTANCE_THRESHOLD = 1e-4
