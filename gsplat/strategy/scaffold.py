from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch

from .base import Strategy
from .ops import duplicate, remove, grow_anchors


@dataclass
class ScaffoldStrategy(Strategy):
    """A neural gaussian strategy that follows the paper:

    `Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering <https://openaccess.thecvf.com/CVPR2024>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small scales.
    - Periodically split GSs with high image plane gradients and large scales.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          split/duplicated. Default is 0.0002.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        refine_every (int): Refine GSs every this steps. Default is 100.
        pause_refine_after_reset (int): Pause refining GSs until this number of steps after
          reset, Default is 0 (no pause at all) and one might want to set this number to the
          number of images in training set.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        revised_opacity (bool): Whether to use revised opacity heuristic from
          arXiv:2404.06109 (experimental). Default is False.
        verbose (bool): Whether to print verbose information. Default is False.

    Examples:

        >>> from gsplat import DefaultStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = DefaultStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    max_voxel_levels: int = 3
    voxel_size: float = 0.001
    refine_stop_iter: int = 15_000
    feat_dim: int = 32
    n_feat_offsets: int = 10
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = True
    colors_mlp: torch.nn.Sequential = torch.nn.Sequential(
        torch.nn.Linear(feat_dim + 3, feat_dim),
        torch.nn.ReLU(True),
        torch.nn.Linear(feat_dim, 3 * n_feat_offsets),
        torch.nn.Sigmoid()
    ).cuda()
    opacities_mlp: torch.nn.Sequential = torch.nn.Sequential(
        torch.nn.Linear(feat_dim + 3, feat_dim),
        torch.nn.ReLU(True),
        torch.nn.Linear(feat_dim, n_feat_offsets),
        torch.nn.Tanh()
    ).cuda()
    scale_rot_mlp: torch.nn.Sequential = torch.nn.Sequential(
        torch.nn.Linear(feat_dim + 3, feat_dim),
        torch.nn.ReLU(True),
        torch.nn.Linear(feat_dim, 7 * n_feat_offsets)
    ).cuda()

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"grad2d": None,
                 "count": None,
                 "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"anchors", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        expected_params = ["anchors",
                           "features",
                           "offsets",
                           "scales",
                           "quats",
                           "opacities_mlp",
                           "colors_mlp",
                           "scale_rot_mlp"]

        assert len(expected_params) == len(params), "expected params and actual params don't match"
        for key in expected_params:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            "means2d" in info
        ), "The 2D anchors of the Gaussians is required but missing."
        info["means2d"].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            # grow GSs
            new_anchors = self._anchor_growing(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {new_anchors} anchors grown."
                    f"Now having {len(params['anchors'])} GSs."
                )

            # prune GSs
            # n_prune = self._prune_gs(params, optimizers, state, step)
            # if self.verbose:
            #     print(
            #         f"Step {step}: {n_prune} GSs pruned. "
            #         f"Now having {len(params['anchors'])} GSs."
            #     )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()

            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in ["width", "height", "n_cameras", "radii", "gaussian_ids"]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = params["anchors"].shape[0]
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian * self.n_feat_offsets, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian * self.n_feat_offsets, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian * self.n_feat_offsets, device=grads.device)


        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel]  # [nnz]
        # update neural gaussian statis
        visible_anchor_mask = info["visible_anchor_mask"]
        neural_selection_mask = info["neural_selection_mask"]
        # Extend to
        anchor_visible_mask = visible_anchor_mask.unsqueeze(dim=1).repeat([1, self.n_feat_offsets]).view(-1)
        neural_gaussian_mask = torch.zeros_like(state["grad2d"], dtype=torch.bool)
        neural_gaussian_mask[anchor_visible_mask] = neural_selection_mask
        valid_mask = neural_gaussian_mask[gs_ids]

        # Filter gs_ids and grads based on the valid_mask
        valid_gs_ids = gs_ids[valid_mask]
        valid_grads_norm = grads.norm(dim=-1)[valid_mask]

        state["grad2d"].index_add_(0, valid_gs_ids, valid_grads_norm)
        state["count"].index_add_(0, valid_gs_ids, torch.ones_like(valid_gs_ids, dtype=torch.float32))

        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _anchor_growing(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int]:
        """
        Implements the Anchor Growing algorithm as described in Algorithm 1 of the
        GS-Scaffold appendix:
        https://openaccess.thecvf.com/content/CVPR2024/supplemental/Lu_Scaffold-GS_Structured_3D_CVPR_2024_supplemental.pdf

        This method performs anchor growing for structured optimization during training,
        which helps improve the generalization and stability of the model.

        Args:
            params (Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict]):
                The model's parameters to be optimized.
            optimizers (Dict[str, torch.optim.Optimizer]):
                A dictionary of optimizers associated with the model's parameters.
            state (Dict[str, Any]):
                A dictionary containing the current state of the training process.
            step (int):
                The current step or iteration of the training process.

        Returns:
            Tuple[int]:
                A tuple containing the updated step value after anchor growing.
        """

        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        grads[grads.isnan()] = 0.0
        device = grads.device

        # is_grad_high = (grads > self.grow_grad2d).squeeze(-1)
        # is_small = (n
        #     torch.exp(params["scales"]).max(dim=-1).values
        #     <= self.grow_scale3d * state["scene_scale"]
        # )
        # is_dupli = is_grad_high & is_small
        # n_dupli = is_dupli.sum().item()

        n_init_features = params["features"].shape[0] * self.n_feat_offsets
        n_added_anchors = 0

        # Algorithm 1: Anchor Growing
        # Step 1: Initialization
        m = 1 # Iteration count (levels)
        M = self.max_voxel_levels
        tau_g = self.grow_grad2d
        epsilon_g = self.voxel_size
        new_anchors: torch.tensor

        # Step 2: Iterate while m <= M
        while m <= M:
            n_feature_diff = params["features"].shape[0] * self.n_feat_offsets - n_init_features
            # Check if anchor candidates have grown 
            if n_feature_diff == 0 and (m-1) > 0:
                break

            # Step 3: Update threshold and voxel size
            tau = tau_g * 2 ** (m - 1)
            current_voxel_size = (16 // (4 ** (m - 1))) * epsilon_g

            # Step 4: Mask from grad threshold. Select neural gaussians (Select candidates)
            gradient_mask = grads >= tau
            gradient_mask = torch.cat([gradient_mask, torch.zeros(n_feature_diff, dtype=torch.bool, device=device)], dim=0)
            neural_gaussians = params["anchors"].unsqueeze(dim=1) + params["offsets"] * torch.exp(params["scales"])[:,:3].unsqueeze(dim=1)
            selected_neural_gaussians = neural_gaussians.view([-1,3])[gradient_mask]

            # Step 5: Merge same positions
            selected_grid_coords = torch.round(selected_neural_gaussians / current_voxel_size).int()
            selected_grid_coords_unique, inv_idx = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            # Step 6: Random elimination
            # TODO: implement rand elimination (necessary)?

            # Get the grid coordinates of the current anchors
            grid_anchor_coords = torch.round(params["anchors"] / current_voxel_size).int()

            # Step 7: Remove occupied by comparing the unique coordinates to current anchors
            remove_occupied_pos_mask = ~(
                (selected_grid_coords_unique.unsqueeze(1) == grid_anchor_coords).all(-1).any(-1).view(-1))

            # New anchor candidates are those unique coordinates that are not duplicates
            new_anchors = selected_grid_coords_unique[remove_occupied_pos_mask]

            if new_anchors.shape[0] > 0:
                grow_anchors(params=params,
                             optimizers=optimizers,
                             state=state,
                             anchors=new_anchors,
                             gradient_mask=gradient_mask,
                             remove_duplicates_mask=remove_occupied_pos_mask,
                             inv_idx=inv_idx,
                             voxel_size=current_voxel_size,
                             n_feat_offsets=self.n_feat_offsets,
                             feat_dim=self.feat_dim)

            n_added_anchors += new_anchors.shape[0]
            m += 1

        return n_added_anchors

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but set `refine_scale2d_stop_iter`
            # to 0 by default to disable it.
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
