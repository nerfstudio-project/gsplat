import warnings

from .rendering import rasterization
from .version import __version__


def spherical_harmonics(*args, **kwargs):
    # Always use new version which can be compatible with the old one.
    from .cuda._wrapper import spherical_harmonics

    if ("method" in kwargs) or (len(args) == 4 and isinstance(args[-1], str)):
        warnings.warn(
            "The 'method' argument is deprecated and will be removed in a future release. ",
            DeprecationWarning,
        )
        args = args[:3]
        kwargs.pop("method", None)
    return spherical_harmonics(*args, **kwargs)


def rasterize_gaussians(*args, **kwargs):
    from .cuda_legacy._wrapper import rasterize_gaussians

    warnings.warn(
        "'rasterize_gaussians is deprecated and will be removed in a future release. "
        "Use gsplat.rasterization for end-to-end rasterizing GSs to images instead.",
        DeprecationWarning,
    )
    return rasterize_gaussians(*args, **kwargs)


def project_gaussians(*args, **kwargs):
    from .cuda_legacy._wrapper import project_gaussians

    warnings.warn(
        "'project_gaussians is deprecated and will be removed in a future release. "
        "Use gsplat.rasterization for end-to-end rasterizing GSs to images instead.",
        DeprecationWarning,
    )
    return project_gaussians(*args, **kwargs)


def map_gaussian_to_intersects(*args, **kwargs):
    from .cuda_legacy._wrapper import map_gaussian_to_intersects

    warnings.warn(
        "'map_gaussian_to_intersects is deprecated and will be removed in a future release. "
        "Use gsplat.rasterization for end-to-end rasterizing GSs to images instead.",
        DeprecationWarning,
    )
    return map_gaussian_to_intersects(*args, **kwargs)


def bin_and_sort_gaussians(*args, **kwargs):
    from .cuda_legacy._wrapper import bin_and_sort_gaussians

    warnings.warn(
        "'bin_and_sort_gaussians is deprecated and will be removed in a future release. "
        "Use gsplat.rasterization for end-to-end rasterizing GSs to images instead.",
        DeprecationWarning,
    )
    return bin_and_sort_gaussians(*args, **kwargs)


def compute_cumulative_intersects(*args, **kwargs):
    from .cuda_legacy._wrapper import compute_cumulative_intersects

    warnings.warn(
        "'compute_cumulative_intersects is deprecated and will be removed in a future release. "
        "Use gsplat.rasterization for end-to-end rasterizing GSs to images instead.",
        DeprecationWarning,
    )
    return compute_cumulative_intersects(*args, **kwargs)


def compute_cov2d_bounds(*args, **kwargs):
    from .cuda_legacy._wrapper import compute_cov2d_bounds

    warnings.warn(
        "'compute_cov2d_bounds is deprecated and will be removed in a future release. "
        "Use gsplat.rasterization for end-to-end rasterizing GSs to images instead.",
        DeprecationWarning,
    )
    return compute_cov2d_bounds(*args, **kwargs)


def get_tile_bin_edges(*args, **kwargs):
    from .cuda_legacy._wrapper import get_tile_bin_edges

    warnings.warn(
        "'get_tile_bin_edges is deprecated and will be removed in a future release. "
        "Use gsplat.rasterization for end-to-end rasterizing GSs to images instead.",
        DeprecationWarning,
    )
    return get_tile_bin_edges(*args, **kwargs)


all = [
    "rasterization",
    "spherical_harmonics",
    "__version__",
    # deprecated
    "rasterize_gaussians",
    "project_gaussians",
    "map_gaussian_to_intersects",
    "bin_and_sort_gaussians",
    "compute_cumulative_intersects",
    "compute_cov2d_bounds",
    "get_tile_bin_edges",
]
