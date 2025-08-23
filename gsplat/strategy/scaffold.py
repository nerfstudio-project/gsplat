from dataclasses import dataclass
from typing import Any, Dict, Union
import torch

from .base import Strategy
from .ops import remove_anchors, grow_anchors
from .ops import relocate
import math


@dataclass
class ScaffoldStrategy(Strategy):
    """A neural gaussian strategy that follows the paper:

    `Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering <https://openaccess.thecvf.com/CVPR2024>`_

    The strategy will:

    - Periodically grows anchors with high image plane gradients.
    - Periodically teleport anchors with low opacity to a place that has high opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): Threshold for pruning GSs with opacity below this value. Default is 0.005.
        grow_grad2d (float): Threshold for splitting/duplicating GSs based on image plane gradient. Default is 0.0002.
        refine_start_iter (int): Iteration to start refining GSs. Default is 500.
        refine_stop_iter (int): Iteration to stop refining GSs. Default is 15,000.
        refine_every (int): Frequency (in steps) at which GSs are refined. Default is 100.
        absgrad (bool): Whether to use absolute gradients for GS splitting. Default is False.
        verbose (bool): If True, prints detailed information during refinement. Default is False.
        max_voxel_levels (int): Maximum levels for voxel splitting during GS growth. Default is 3.
        voxel_size (float): Base size of the voxel used in GS growth. Default is 0.001.
        pruning_thresholds (float): Threshold for pruning based on refinement steps. Default is 0.8.
        growing_thresholds (float): Threshold for GS growth based on refinement steps. Default is 0.4.
        drop_out (bool): If True, applies dropout during GS growth to prevent overgrowth. Default is False.
        pruning (bool): If True, enables pruning of GSs during refinement. Default is False.

    Examples:

        >>> from gsplat import ScaffoldStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = ScaffoldStrategy()
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
    grow_grad2d: float = 1.28e-4
    refine_start_iter: int = 800
    absgrad: bool = False
    max_voxel_levels: int = 3
    voxel_size: float = 0.001
    refine_stop_iter: int = 15_000
    refine_every: int = 100

    # 3.3 Observation Thresholds (compare paper)
    # "To enhance the robustness of the Growing and Pruning operations for long image sequences, ..."
    pruning_thresholds: float = 0.8
    growing_thresholds: float = 0.4

    verbose: bool = True
    drop_out: bool = False
    pruning: bool = False

    def initialize_state(
        self, scene_scale: float = 1.0, feat_dim=128, n_feat_offsets=10
    ) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.

        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        state = {
            "binoms": binoms,
            "grad2d": None,
            "count": None,
            "anchor_count": None,
            "anchor_opacity": None,
            "scene_scale": scene_scale,
            "feat_dim": feat_dim,
            "n_feat_offsets": n_feat_offsets,
        }
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
        expected_params = [
            "anchors",
            "features",
            "offsets",
            "scales",
            "quats",
            "opacities",
        ]

        assert len(expected_params) == len(
            params
        ), "expected params and actual params don't match"
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

        # move to the correct device
        state["binoms"] = state["binoms"].to(params["anchors"].device)
        binoms = state["binoms"]

        if step > 500:
            self._update_state(params, state, info, packed=packed)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            # grow GSs
            new_anchors = self._anchor_growing(params, optimizers, state)
            if self.verbose:
                print(
                    f"Step {step}: {new_anchors} anchors grown. Now having {len(params['anchors'])} anchors."
                )

            if self.pruning:
                low_opacity_mask = (
                    state["anchor_opacity"] < self.prune_opa * state["anchor_count"]
                ).squeeze()
                anchor_mask = (
                    state["anchor_count"] > self.pruning_thresholds * self.refine_every
                )  # [N, 1]
                is_prune = torch.logical_and(low_opacity_mask, anchor_mask)

                indices = is_prune.nonzero(as_tuple=False).squeeze()
                # Efficiently set the specified indices to zero
                state["anchor_count"].index_fill_(0, indices, 0)
                state["anchor_opacity"].index_fill_(0, indices, 0)

                n_prune = is_prune.sum().item()
                if n_prune > 0:
                    remove_anchors(
                        params=params,
                        optimizers=optimizers,
                        n_feat_offsets=state["n_feat_offsets"],
                        state=state,
                        mask=is_prune,
                    )

                if self.verbose:
                    print(
                        f"Step {step}: {n_prune} anchors pruned. Now having {len(params['anchors'])} anchors."
                    )
            else:
                n_relocated_gs = self._relocate_gs(params, optimizers, binoms, state)
                if self.verbose:
                    print(f"Relocated anchors {n_relocated_gs}")

            torch.cuda.empty_cache()

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        # Ensure required keys are present
        required_keys = ["width", "height", "n_cameras", "radii", "gaussian_ids"]
        for key in required_keys:
            assert key in info, f"{key} is required but missing."

        # Normalize gradients to [-1, 1] screen space
        factor = 0.5 * info["n_cameras"]
        scale_factors = torch.tensor(
            [info["width"] * factor, info["height"] * factor],
            device=info["means2d"].device,
        )

        if self.absgrad:
            grads = info["means2d"].absgrad.detach() * scale_factors
        else:
            grads = info["means2d"].grad.detach() * scale_factors

        # Initialize state on the first run
        n_gaussian = params["anchors"].shape[0]
        device = grads.device
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(
                n_gaussian * state["n_feat_offsets"], device=device
            )
        if state["count"] is None:
            state["count"] = torch.zeros(
                n_gaussian * state["n_feat_offsets"], device=device
            )
        if state["anchor_count"] is None:
            state["anchor_count"] = torch.zeros(n_gaussian, device=device)
        if state["anchor_opacity"] is None:
            state["anchor_opacity"] = torch.zeros(n_gaussian, device=device)

        neural_selection_mask = info["neural_selection_mask"]

        # Update the running state
        sel = info["radii"] > 0.0  # [C, N]
        neural_ids = neural_selection_mask.nonzero(as_tuple=False).squeeze(-1)
        grads = grads[sel].norm(dim=-1)  # [nnz, 2]
        sel = sel.squeeze(0)

        valid_ids = neural_ids[sel]

        # Update state using index_add_
        state["grad2d"].index_add_(0, valid_ids, grads)
        state["count"].index_add_(
            0,
            valid_ids,
            torch.ones((valid_ids.shape[0]), dtype=torch.float32, device=device),
        )

        # Update anchor opacity and count
        visible_anchor_mask = info["visible_anchor_mask"]
        anchor_ids = visible_anchor_mask.nonzero(as_tuple=False).squeeze(-1)
        neural_opacities = (
            info["neural_opacities"]
            .detach()
            .view(-1, state["n_feat_offsets"])
            .clamp_min_(0)
            .sum(dim=1)
        )

        state["anchor_opacity"].index_add_(0, anchor_ids, neural_opacities)
        state["anchor_count"].index_add_(
            0, anchor_ids, torch.ones_like(anchor_ids, dtype=torch.float32)
        )

    @torch.no_grad()
    def _anchor_growing(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ) -> int:
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

        Returns:
            int:
                Number of growned anchors.
        """

        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        grads[grads.isnan()] = 0.0
        device = grads.device

        n_init_features = state["count"].shape[0]
        n_added_anchors = 0

        # Algorithm 1: Anchor Growing
        # Step 1: Initialization
        m = 1  # Iteration count (levels)
        M = self.max_voxel_levels
        tau_g = self.grow_grad2d
        epsilon_g = self.voxel_size
        new_anchors: torch.tensor

        growing_threshold_mask = (
            state["count"] > self.refine_every * self.growing_thresholds
        )
        # Step 2: Iterate while m <= M
        while m <= M:
            n_feature_diff = state["count"].shape[0] - n_init_features
            # Check if anchor candidates have grown
            if n_feature_diff == 0 and m > 1:
                break

            # Step 3: Update threshold and voxel size
            tau = tau_g * (2 ** (m - 1))
            current_voxel_size = (16 // (4 ** (m - 1))) * epsilon_g

            # Step 4: Mask from grad threshold. Select neural gaussians (Select candidates)
            gradient_mask = grads >= tau
            gradient_mask = torch.logical_and(gradient_mask, growing_threshold_mask)

            # Drop-out: Helps prevent too massive anchor growth.
            if self.drop_out:
                rand_mask = torch.rand_like(gradient_mask.float()) > (0.5**m)
                rand_mask = rand_mask.cuda()
                gradient_mask = torch.logical_and(gradient_mask, rand_mask)

            gradient_mask = torch.cat(
                [
                    gradient_mask,
                    torch.zeros(n_feature_diff, dtype=torch.bool, device=device),
                ],
                dim=0,
            )

            # Compute neural gaussians
            neural_gaussians = params["anchors"].unsqueeze(dim=1) + params[
                "offsets"
            ] * torch.exp(params["scales"][:, :3]).unsqueeze(dim=1)
            selected_neural_gaussians = neural_gaussians.view([-1, 3])[gradient_mask]

            # Step 5: Merge same positions
            selected_grid_coords = torch.round(
                selected_neural_gaussians / current_voxel_size
            ).int()
            selected_grid_coords_unique, inv_idx = torch.unique(
                selected_grid_coords, return_inverse=True, dim=0
            )

            # Get the grid coordinates of the current anchors
            grid_anchor_coords = torch.round(
                params["anchors"] / current_voxel_size
            ).int()

            # Step 6: Remove occupied by comparing the unique coordinates to current anchors
            def coords_to_indices(coords, N):
                """
                Maps quantized multi-dimensional coordinates to unique 1D indices without using torch.matmul.

                Args:
                    coords (torch.Tensor): Tensor of shape [num_points, D], where D is the dimensionality.
                    N (int): A large enough number to ensure uniqueness of indices.

                Returns:
                    torch.Tensor: Tensor of unique indices corresponding to the coordinates.
                """
                D = coords.shape[1]
                device = coords.device
                dtype = coords.dtype  # Keep the original data type

                # Compute N_powers as integers
                N_powers = N ** torch.arange(D - 1, -1, -1, device=device, dtype=dtype)

                # Perform element-wise multiplication and sum along the last dimension
                indices = (coords * N_powers).sum(dim=1)

                return indices

            #  Quantize the coordinates
            decimal_places = 6  # precision
            scale = 10**decimal_places

            # Quantize the coordinates by scaling and converting to integers
            selected_coords_quant = (selected_grid_coords_unique * scale).long()
            anchor_coords_quant = (grid_anchor_coords * scale).long()

            # Compute the maximum coordinate value
            max_coord_value = (
                torch.max(torch.cat([selected_coords_quant, anchor_coords_quant])) + 1
            )
            N = max_coord_value.item()

            # Compute unique indices for both coordinate sets
            indices_selected = coords_to_indices(selected_coords_quant, N)
            indices_anchor = coords_to_indices(anchor_coords_quant, N)

            remove_occupied_pos_mask = ~torch.isin(indices_selected, indices_anchor)

            # New anchor candidates are those unique coordinates that are not duplicates
            new_anchors = selected_grid_coords_unique[remove_occupied_pos_mask]

            if new_anchors.shape[0] > 0:
                grow_anchors(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    anchors=new_anchors,
                    gradient_mask=gradient_mask,
                    remove_duplicates_mask=remove_occupied_pos_mask,
                    inv_idx=inv_idx,
                    voxel_size=current_voxel_size,
                    n_feat_offsets=state["n_feat_offsets"],
                    feat_dim=state["feat_dim"],
                )

            n_added_anchors += new_anchors.shape[0]
            m += 1

        indices = torch.arange(n_init_features, device=growing_threshold_mask.device)[
            growing_threshold_mask
        ]

        if indices.numel() > 0:
            state["count"].index_fill_(0, indices, 0)
            state["grad2d"].index_fill_(0, indices, 0)

        return n_added_anchors

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: torch.Tensor,
        state: Dict[str, Any],
    ) -> int:
        dead_mask = (
            state["anchor_opacity"] < self.prune_opa * state["anchor_count"]
        ).squeeze()
        n_gs = dead_mask.sum().item()
        if n_gs > 0:
            relocate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=dead_mask,
                binoms=binoms,
                min_opacity=self.prune_opa,
            )
        return n_gs
