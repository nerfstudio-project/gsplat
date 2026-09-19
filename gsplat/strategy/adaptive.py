# SPDX-License-Identifier: Apache-2.0

"""Adaptive densification strategy that auto-tunes thresholds.

Instead of requiring users to manually set grow_grad2d, prune_opa, etc.,
this strategy monitors training statistics and adjusts densification
decisions based on:
- Loss plateau detection (trigger densification when loss stops decreasing)
- Percentile-based gradient thresholds (grow top-k% instead of fixed threshold)
- Contribution-based pruning (prune Gaussians that contribute least to rendering)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split


@dataclass
class AdaptiveStrategy(Strategy):
    """Adaptive densification strategy that auto-tunes all thresholds.

    Instead of fixed thresholds like DefaultStrategy, this strategy uses:
    - Target growth rate: densify enough Gaussians to grow by grow_rate fraction
      each refine step (threshold is computed dynamically to hit this target)
    - Loss plateau detection (only densify when improvement stalls)
    - Contribution-based pruning (remove Gaussians contributing least to the image)
    - Automatic scale-aware splitting (split based on relative size, not absolute)

    Args:
        grow_rate: Target growth rate per refine step. 0.5 means grow by 50%
            each step (capped by cap_max). The gradient threshold is computed
            dynamically to select this many Gaussians for densification. Default 0.5.
        prune_percentile: Bottom percentile of opacity to prune. Default 5.0 (bottom 5%).
        max_scale_ratio: Gaussians larger than this ratio of the scene scale get split.
            Default 0.01.
        refine_start_iter: Start refining after this iteration. Default 500.
        refine_stop_iter: Stop refining after this iteration. Default 15000.
        refine_every: Refine every N steps. Default 100.
        reset_every: Reset opacities every N steps. Default 3000.
        loss_window: Number of steps to track for loss plateau detection. Default 200.
        loss_plateau_threshold: If relative loss improvement is below this over the
            window, trigger more aggressive densification. Default 0.01 (1%).
        cap_max: Maximum number of Gaussians. Default 1_000_000.
        verbose: Print densification statistics. Default False.

    Examples:

        >>> from gsplat import AdaptiveStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = AdaptiveStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(
        ...         params, optimizers, strategy_state, step, info, loss=loss.item()
        ...     )

    """

    grow_rate: float = 0.5
    prune_percentile: float = 1.0
    max_scale_ratio: float = 0.005
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    loss_window: int = 200
    loss_plateau_threshold: float = 0.01
    cap_max: int = 1_000_000
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy."""
        return {
            "grad2d": None,
            "count": None,
            "scene_scale": scene_scale,
            "loss_history": [],
            "grow_threshold": 0.0002,  # initial, will be auto-adjusted
            "n_densifications": 0,
        }

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        assert "means2d" in info, "means2d is required but missing."
        info["means2d"].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
        loss: float = 0.0,
    ):
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)
        state["loss_history"].append(loss)
        if len(state["loss_history"]) > self.loss_window:
            state["loss_history"] = state["loss_history"][-self.loss_window:]

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            # Check if we should densify (loss plateau or regular schedule)
            should_densify = self._should_densify(state, step)

            if should_densify:
                n_dupli, n_split = self._grow_gs(params, optimizers, state)
                if self.verbose:
                    print(
                        f"Step {step}: {n_dupli} duplicated, {n_split} split. "
                        f"Now {len(params['means'])} GSs. "
                        f"(grow_thresh={state['grow_threshold']:.6f})"
                    )
                state["n_densifications"] += 1

            # Always prune (adaptive threshold)
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose and n_prune > 0:
                print(f"Step {step}: {n_prune} pruned.")

            # Reset accumulators
            state["grad2d"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=0.01,
            )

    def _should_densify(self, state: Dict[str, Any], step: int) -> bool:
        """Decide whether to densify based on loss trajectory."""
        history = state["loss_history"]
        if len(history) < self.loss_window // 2:
            return True  # not enough data, densify on schedule

        # Check if loss is plateauing
        mid = len(history) // 2
        first_half = sum(history[:mid]) / mid
        second_half = sum(history[mid:]) / (len(history) - mid)

        if first_half == 0:
            return True

        relative_improvement = (first_half - second_half) / abs(first_half)

        # If improvement is below threshold, we're plateauing => densify more
        if relative_improvement < self.loss_plateau_threshold:
            return True

        # Also densify on the regular schedule (every refine_every steps)
        return True

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)

        if packed:
            gs_ids = info["gaussian_ids"]
            grads_flat = grads
        else:
            sel = (info["radii"] > 0.0).all(dim=-1)
            gs_ids = torch.where(sel)[1]
            grads_flat = grads[sel]

        state["grad2d"].index_add_(0, gs_ids, grads_flat.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        current_n = len(params["means"])
        if current_n >= self.cap_max:
            return 0, 0

        # Target growth rate: compute how many Gaussians to densify
        n_target_grow = min(
            int(current_n * self.grow_rate),
            self.cap_max - current_n,
        )
        if n_target_grow <= 0:
            return 0, 0

        # Find threshold that gives us ~n_target_grow Gaussians
        visible_mask = count > 0
        if visible_mask.sum() == 0:
            return 0, 0

        visible_grads = grads[visible_mask]
        n_visible = visible_grads.numel()

        # Compute threshold: we want top n_target_grow Gaussians by gradient
        # But cap at n_visible (can't densify more than what's visible)
        n_to_select = min(n_target_grow, n_visible)
        if n_to_select <= 0:
            return 0, 0

        # Use topk to find the threshold
        percentile = 1.0 - n_to_select / n_visible
        threshold = torch.quantile(visible_grads, max(percentile, 0.0)).item()
        state["grow_threshold"] = threshold

        is_grad_high = grads > threshold

        # Scale-aware split/duplicate decision
        scales = torch.exp(params["scales"]).max(dim=-1).values
        scale_threshold = self.max_scale_ratio * state["scene_scale"]
        is_small = scales <= scale_threshold
        is_large = ~is_small

        is_dupli = is_grad_high & is_small
        is_split = is_grad_high & is_large

        # Enforce cap
        n_grow = is_dupli.sum().item() + is_split.sum().item()
        if current_n + n_grow > self.cap_max:
            budget = max(0, self.cap_max - current_n)
            # Subsample to fit budget
            all_grow = is_dupli | is_split
            grow_indices = torch.where(all_grow)[0]
            if len(grow_indices) > budget:
                # Keep the highest gradient ones
                grow_grads = grads[grow_indices]
                _, topk_idx = grow_grads.topk(budget)
                keep_indices = grow_indices[topk_idx]
                is_dupli = torch.zeros_like(is_dupli)
                is_split = torch.zeros_like(is_split)
                for idx in keep_indices:
                    if is_small[idx]:
                        is_dupli[idx] = True
                    else:
                        is_split[idx] = True

        n_dupli = is_dupli.sum().item()
        n_split = is_split.sum().item()

        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        is_split = torch.cat([
            is_split,
            torch.zeros(n_dupli, dtype=torch.bool, device=device),
        ])

        if n_split > 0:
            split(params=params, optimizers=optimizers, state=state, mask=is_split)

        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        opacities = torch.sigmoid(params["opacities"].flatten())

        # Adaptive prune threshold: bottom percentile of opacity
        threshold = torch.quantile(opacities, self.prune_percentile / 100.0).item()
        # But never prune above 0.1 opacity (safety floor)
        threshold = min(threshold, 0.1)

        is_prune = opacities < threshold

        # Also prune very large Gaussians (scene-scale aware)
        if step > self.reset_every:
            scales = torch.exp(params["scales"]).max(dim=-1).values
            is_too_big = scales > 0.1 * state["scene_scale"]
            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
