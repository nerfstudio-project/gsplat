# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import colorsys
import dataclasses
import imageio
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal
from jaxtyping import Float
import torch
from torch import Tensor
from rich.console import Console
import numpy as np
import splines
import splines.quaternion
import viser
import viser.transforms as tf
from scipy import interpolate
import matplotlib


@dataclasses.dataclass
class Keyframe:
    position: np.ndarray
    wxyz: np.ndarray
    override_fov_enabled: bool
    override_fov_rad: float
    override_time_enabled: bool
    override_time_val: float
    aspect: float
    override_transition_enabled: bool
    override_transition_sec: Optional[float]

    @staticmethod
    def from_camera(camera: viser.CameraHandle, aspect: float) -> Keyframe:
        return Keyframe(
            camera.position,
            camera.wxyz,
            override_fov_enabled=False,
            override_fov_rad=camera.fov,
            override_time_enabled=False,
            override_time_val=0.0,
            aspect=aspect,
            override_transition_enabled=False,
            override_transition_sec=None,
        )


class CameraPath:
    def __init__(
        self,
        server: viser.ViserServer,
        duration_element: viser.GuiInputHandle[float],
        time_enabled: bool = False,
    ):
        self._server = server
        self._keyframes: Dict[int, Tuple[Keyframe, viser.CameraFrustumHandle]] = {}
        self._keyframe_counter: int = 0
        self._spline_nodes: List[viser.SceneNodeHandle] = []
        self._camera_edit_panel: Optional[viser.Gui3dContainerHandle] = None

        self._orientation_spline: Optional[splines.quaternion.KochanekBartels] = None
        self._position_spline: Optional[splines.KochanekBartels] = None
        self._fov_spline: Optional[splines.KochanekBartels] = None
        self._keyframes_visible: bool = True

        self._duration_element = duration_element

        # These parameters should be overridden externally.
        self.loop: bool = False
        self.framerate: float = 30.0
        self.tension: float = 0.5  # Tension / alpha term.
        self.default_fov: float = 0.0
        self.time_enabled = time_enabled
        self.default_render_time: float = 0.0
        self.default_transition_sec: float = 0.0
        self.show_spline: bool = True

    def set_keyframes_visible(self, visible: bool) -> None:
        self._keyframes_visible = visible
        for keyframe in self._keyframes.values():
            keyframe[1].visible = visible

    def add_camera(
        self, keyframe: Keyframe, keyframe_index: Optional[int] = None
    ) -> None:
        """Add a new camera, or replace an old one if `keyframe_index` is passed in."""
        server = self._server

        # Add a keyframe if we aren't replacing an existing one.
        if keyframe_index is None:
            keyframe_index = self._keyframe_counter
            self._keyframe_counter += 1

        frustum_handle = server.scene.add_camera_frustum(
            f"/render_cameras/{keyframe_index}",
            fov=(
                keyframe.override_fov_rad
                if keyframe.override_fov_enabled
                else self.default_fov
            ),
            aspect=keyframe.aspect,
            scale=0.1,
            color=(200, 10, 30),
            wxyz=keyframe.wxyz,
            position=keyframe.position,
            visible=self._keyframes_visible,
        )
        self._server.scene.add_icosphere(
            f"/render_cameras/{keyframe_index}/sphere",
            radius=0.03,
            color=(200, 10, 30),
        )

        @frustum_handle.on_click
        def _(_) -> None:
            if self._camera_edit_panel is not None:
                self._camera_edit_panel.remove()
                self._camera_edit_panel = None

            with server.scene.add_3d_gui_container(
                "/camera_edit_panel",
                position=keyframe.position,
            ) as camera_edit_panel:
                self._camera_edit_panel = camera_edit_panel
                override_fov = server.gui.add_checkbox(
                    "Override FOV", initial_value=keyframe.override_fov_enabled
                )
                override_fov_degrees_slider = server.gui.add_slider(
                    "Override FOV (degrees)",
                    5.0,
                    175.0,
                    step=0.1,
                    initial_value=keyframe.override_fov_rad * 180.0 / np.pi,
                    disabled=not keyframe.override_fov_enabled,
                )
                if self.time_enabled:
                    override_time = server.gui.add_checkbox(
                        "Override Time", initial_value=keyframe.override_time_enabled
                    )
                    override_time_val = server.gui.add_slider(
                        "Override Time",
                        0.0,
                        1.0,
                        step=0.01,
                        initial_value=keyframe.override_time_val,
                        disabled=not keyframe.override_time_enabled,
                    )

                    @override_time.on_update
                    def _(_) -> None:
                        keyframe.override_time_enabled = override_time.value
                        override_time_val.disabled = not override_time.value
                        self.add_camera(keyframe, keyframe_index)

                    @override_time_val.on_update
                    def _(_) -> None:
                        keyframe.override_time_val = override_time_val.value
                        self.add_camera(keyframe, keyframe_index)

                delete_button = server.gui.add_button(
                    "Delete", color="red", icon=viser.Icon.TRASH
                )
                go_to_button = server.gui.add_button("Go to")
                close_button = server.gui.add_button("Close")

            @override_fov.on_update
            def _(_) -> None:
                keyframe.override_fov_enabled = override_fov.value
                override_fov_degrees_slider.disabled = not override_fov.value
                self.add_camera(keyframe, keyframe_index)

            @override_fov_degrees_slider.on_update
            def _(_) -> None:
                keyframe.override_fov_rad = (
                    override_fov_degrees_slider.value / 180.0 * np.pi
                )
                self.add_camera(keyframe, keyframe_index)

            @delete_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                with event.client.gui.add_modal("Confirm") as modal:
                    event.client.gui.add_markdown("Delete keyframe?")
                    confirm_button = event.client.gui.add_button(
                        "Yes", color="red", icon=viser.Icon.TRASH
                    )
                    exit_button = event.client.gui.add_button("Cancel")

                    @confirm_button.on_click
                    def _(_) -> None:
                        assert camera_edit_panel is not None

                        keyframe_id = None
                        for i, keyframe_tuple in self._keyframes.items():
                            if keyframe_tuple[1] is frustum_handle:
                                keyframe_id = i
                                break
                        assert keyframe_id is not None

                        self._keyframes.pop(keyframe_id)
                        frustum_handle.remove()
                        camera_edit_panel.remove()
                        self._camera_edit_panel = None
                        modal.close()
                        self.update_spline()

                    @exit_button.on_click
                    def _(_) -> None:
                        modal.close()

            @go_to_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                client = event.client
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )
                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(keyframe.wxyz), keyframe.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(10):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 9.0
                    )

                    # Important bit: we atomically set both the orientation and the position
                    # of the camera.
                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                    time.sleep(1.0 / 30.0)

            @close_button.on_click
            def _(_) -> None:
                assert camera_edit_panel is not None
                camera_edit_panel.remove()
                self._camera_edit_panel = None

        self._keyframes[keyframe_index] = (keyframe, frustum_handle)

    def update_aspect(self, aspect: float) -> None:
        for keyframe_index, frame in self._keyframes.items():
            frame = dataclasses.replace(frame[0], aspect=aspect)
            self.add_camera(frame, keyframe_index=keyframe_index)

    def get_aspect(self) -> float:
        """Get W/H aspect ratio, which is shared across all keyframes."""
        assert len(self._keyframes) > 0
        return next(iter(self._keyframes.values()))[0].aspect

    def reset(self) -> None:
        for frame in self._keyframes.values():
            frame[1].remove()
        self._keyframes.clear()
        self.update_spline()

    def spline_t_from_t_sec(self, time: np.ndarray) -> np.ndarray:
        """From a time value in seconds, compute a t value for our geometric
        spline interpolation. An increment of 1 for the latter will move the
        camera forward by one keyframe.

        We use a PCHIP spline here to guarantee monotonicity.
        """
        transition_times_cumsum = self.compute_transition_times_cumsum()
        spline_indices = np.arange(transition_times_cumsum.shape[0])

        if self.loop:
            # In the case of a loop, we pad the spline to match the start/end
            # slopes.
            interpolator = interpolate.PchipInterpolator(
                x=np.concatenate(
                    [
                        [-(transition_times_cumsum[-1] - transition_times_cumsum[-2])],
                        transition_times_cumsum,
                        transition_times_cumsum[-1:] + transition_times_cumsum[1:2],
                    ],
                    axis=0,
                ),
                y=np.concatenate(
                    [[-1], spline_indices, [spline_indices[-1] + 1]], axis=0
                ),
            )
        else:
            interpolator = interpolate.PchipInterpolator(
                x=transition_times_cumsum, y=spline_indices
            )

        # Clip to account for floating point error.
        return np.clip(interpolator(time), 0, spline_indices[-1])

    def interpolate_pose_and_fov_rad(
        self, normalized_t: float
    ) -> Optional[Union[Tuple[tf.SE3, float], Tuple[tf.SE3, float, float]]]:
        if len(self._keyframes) < 2:
            return None

        self._fov_spline = splines.KochanekBartels(
            [
                (
                    keyframe[0].override_fov_rad
                    if keyframe[0].override_fov_enabled
                    else self.default_fov
                )
                for keyframe in self._keyframes.values()
            ],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )

        self._time_spline = splines.KochanekBartels(
            [
                (
                    keyframe[0].override_time_val
                    if keyframe[0].override_time_enabled
                    else self.default_render_time
                )
                for keyframe in self._keyframes.values()
            ],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )

        assert self._orientation_spline is not None
        assert self._position_spline is not None
        assert self._fov_spline is not None
        if self.time_enabled:
            assert self._time_spline is not None
        max_t = self.compute_duration()
        t = max_t * normalized_t
        spline_t = float(self.spline_t_from_t_sec(np.array(t)))

        quat = self._orientation_spline.evaluate(spline_t)
        assert isinstance(quat, splines.quaternion.UnitQuaternion)
        if self.time_enabled:
            return (
                tf.SE3.from_rotation_and_translation(
                    tf.SO3(np.array([quat.scalar, *quat.vector])),
                    self._position_spline.evaluate(spline_t),
                ),
                float(self._fov_spline.evaluate(spline_t)),
                float(self._time_spline.evaluate(spline_t)),
            )
        else:
            return (
                tf.SE3.from_rotation_and_translation(
                    tf.SO3(np.array([quat.scalar, *quat.vector])),
                    self._position_spline.evaluate(spline_t),
                ),
                float(self._fov_spline.evaluate(spline_t)),
            )

    def update_spline(self) -> None:
        num_frames = int(self.compute_duration() * self.framerate)
        keyframes = list(self._keyframes.values())

        if num_frames <= 0 or not self.show_spline or len(keyframes) < 2:
            for node in self._spline_nodes:
                node.remove()
            self._spline_nodes.clear()
            return

        transition_times_cumsum = self.compute_transition_times_cumsum()

        self._orientation_spline = splines.quaternion.KochanekBartels(
            [
                splines.quaternion.UnitQuaternion.from_unit_xyzw(
                    np.roll(keyframe[0].wxyz, shift=-1)
                )
                for keyframe in keyframes
            ],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )
        self._position_spline = splines.KochanekBartels(
            [keyframe[0].position for keyframe in keyframes],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )

        # Update visualized spline.
        points_array = self._position_spline.evaluate(
            self.spline_t_from_t_sec(
                np.linspace(0, transition_times_cumsum[-1], num_frames)
            )
        )
        colors_array = np.array(
            [
                colorsys.hls_to_rgb(h, 0.5, 1.0)
                for h in np.linspace(0.0, 1.0, len(points_array))
            ]
        )

        # Clear prior spline nodes.
        for node in self._spline_nodes:
            node.remove()
        self._spline_nodes.clear()

        self._spline_nodes.append(
            self._server.scene.add_spline_catmull_rom(
                "/render_camera_spline",
                positions=points_array,
                color=(220, 220, 220),
                closed=self.loop,
                line_width=1.0,
                segments=points_array.shape[0] + 1,
            )
        )
        self._spline_nodes.append(
            self._server.scene.add_point_cloud(
                "/render_camera_spline/points",
                points=points_array,
                colors=colors_array,
                point_size=0.04,
            )
        )

        def make_transition_handle(i: int) -> None:
            assert self._position_spline is not None
            transition_pos = self._position_spline.evaluate(
                float(
                    self.spline_t_from_t_sec(
                        (transition_times_cumsum[i] + transition_times_cumsum[i + 1])
                        / 2.0,
                    )
                )
            )
            transition_sphere = self._server.scene.add_icosphere(
                f"/render_camera_spline/transition_{i}",
                radius=0.04,
                color=(255, 0, 0),
                position=transition_pos,
            )
            self._spline_nodes.append(transition_sphere)

            @transition_sphere.on_click
            def _(_) -> None:
                server = self._server

                if self._camera_edit_panel is not None:
                    self._camera_edit_panel.remove()
                    self._camera_edit_panel = None

                keyframe_index = (i + 1) % len(self._keyframes)
                keyframe = keyframes[keyframe_index][0]

                with server.scene.add_3d_gui_container(
                    "/camera_edit_panel",
                    position=transition_pos,
                ) as camera_edit_panel:
                    self._camera_edit_panel = camera_edit_panel
                    override_transition_enabled = server.gui.add_checkbox(
                        "Override transition",
                        initial_value=keyframe.override_transition_enabled,
                    )
                    override_transition_sec = server.gui.add_number(
                        "Override transition (sec)",
                        initial_value=(
                            keyframe.override_transition_sec
                            if keyframe.override_transition_sec is not None
                            else self.default_transition_sec
                        ),
                        min=0.001,
                        max=30.0,
                        step=0.001,
                        disabled=not override_transition_enabled.value,
                    )
                    close_button = server.gui.add_button("Close")

                @override_transition_enabled.on_update
                def _(_) -> None:
                    keyframe.override_transition_enabled = (
                        override_transition_enabled.value
                    )
                    override_transition_sec.disabled = (
                        not override_transition_enabled.value
                    )
                    self._duration_element.value = self.compute_duration()

                @override_transition_sec.on_update
                def _(_) -> None:
                    keyframe.override_transition_sec = override_transition_sec.value
                    self._duration_element.value = self.compute_duration()

                @close_button.on_click
                def _(_) -> None:
                    assert camera_edit_panel is not None
                    camera_edit_panel.remove()
                    self._camera_edit_panel = None

        (num_transitions_plus_1,) = transition_times_cumsum.shape
        for i in range(num_transitions_plus_1 - 1):
            make_transition_handle(i)

        # for i in range(transition_times.shape[0])

    def compute_duration(self) -> float:
        """Compute the total duration of the trajectory."""
        total = 0.0
        for i, (keyframe, frustum) in enumerate(self._keyframes.values()):
            if i == 0 and not self.loop:
                continue
            del frustum
            total += (
                keyframe.override_transition_sec
                if keyframe.override_transition_enabled
                and keyframe.override_transition_sec is not None
                else self.default_transition_sec
            )
        return total

    def compute_transition_times_cumsum(self) -> np.ndarray:
        """Compute the total duration of the trajectory."""
        total = 0.0
        out = [0.0]
        for i, (keyframe, frustum) in enumerate(self._keyframes.values()):
            if i == 0:
                continue
            del frustum
            total += (
                keyframe.override_transition_sec
                if keyframe.override_transition_enabled
                and keyframe.override_transition_sec is not None
                else self.default_transition_sec
            )
            out.append(total)

        if self.loop:
            keyframe = next(iter(self._keyframes.values()))[0]
            total += (
                keyframe.override_transition_sec
                if keyframe.override_transition_enabled
                and keyframe.override_transition_sec is not None
                else self.default_transition_sec
            )
            out.append(total)

        return np.array(out)


@dataclasses.dataclass
class RenderTabState:
    """Useful GUI handles exposed by the render tab."""

    num_train_rays_per_sec: Optional[float] = None
    num_view_rays_per_sec: float = 100000.0
    preview_render: bool = False
    preview_fov: float = 0.0
    preview_time: float = 0.0
    preview_aspect: float = 1.0
    viewer_res: int = 2048
    viewer_width: int = 1280
    viewer_height: int = 960
    render_width: int = 1280
    render_height: int = 960


Colormaps = Literal["turbo", "viridis", "magma", "inferno", "cividis", "gray"]


def apply_float_colormap(
    image: Float[Tensor, "*bs 1"], colormap: Colormaps = "viridis"
) -> Float[Tensor, "*bs rgb=3"]:
    """Copied from nerfstudio/utils/colormaps.py
    Convert single channel to a color image.

    Args:
        image: Single channel image.
        colormap: Colormap for image.

    Returns:
        Tensor: Colored image with colors in [0, 1]
    """

    image = torch.nan_to_num(image, 0)
    if colormap == "gray":
        return image.repeat(1, 1, 3)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[
        image_long[..., 0]
    ]


def populate_general_render_tab(
    server: viser.ViserServer,
    output_dir: Path,
    folder: viser.GuiFolderHandle,
    render_tab_state: RenderTabState,
    extra_handles: Optional[Dict[str, viser.GuiInputHandle]] = None,
    scale_ratio: float = 10.0,  # VISER_NERFSTUDIO_SCALE_RATIO
    time_enabled: bool = False,
) -> Dict[str, viser.GuiInputHandle]:
    """
    Populate the render tab with general controls.
    Args:
        server: The server to populate the render tab on.
        output_dir: The path to the output folder.
        folder: The folder to populate the render tab on.
        render_tab_state: The render tab state exposed to the outer scope.
        extra_handles: Extra handles needed to be disabled during dump_video.
        scale_ratio: The scale ratio for the render tab.
        time_enabled: Whether to enable the time slider.
    Returns:
        A dictionary of handles populated in the render tab.
    """
    with folder:
        fov_degrees_slider = server.gui.add_slider(
            "FOV",
            initial_value=50.0,
            min=0.1,
            max=175.0,
            step=0.01,
            hint="Field-of-view for rendering, which can also be overridden on a per-keyframe basis.",
        )

        render_time = None
        if time_enabled:
            render_time = server.gui.add_slider(
                "Default Time",
                initial_value=0.0,
                min=0.0,
                max=1.0,
                step=0.01,
                hint="Rendering time step, which can also be overridden on a per-keyframe basis.",
            )

            @render_time.on_update
            def _(_) -> None:
                camera_path.default_render_time = render_time.value

        @fov_degrees_slider.on_update
        def _(_) -> None:
            fov_radians = fov_degrees_slider.value / 180.0 * np.pi
            for client in server.get_clients().values():
                client.camera.fov = fov_radians
            camera_path.default_fov = fov_radians

            # Updating the aspect ratio will also re-render the camera frustums.
            # Could rethink this.
            camera_path.update_aspect(
                render_res_vec2.value[0] / render_res_vec2.value[1]
            )
            compute_and_update_preview_camera_state()

        render_res_vec2 = server.gui.add_vector2(
            "Render Res",
            initial_value=(1280, 960),
            min=(50, 50),
            max=(10_000, 10_000),
            step=1,
            hint="Rendering resolution.",
        )

        @render_res_vec2.on_update
        def _(_) -> None:
            camera_path.update_aspect(
                render_res_vec2.value[0] / render_res_vec2.value[1]
            )
            compute_and_update_preview_camera_state()
            render_tab_state.render_width = int(render_res_vec2.value[0])
            render_tab_state.render_height = int(render_res_vec2.value[1])

        add_keyframe_button = server.gui.add_button(
            "Add Keyframe",
            icon=viser.Icon.PLUS,
            hint="Add a new keyframe at the current pose.",
        )

        @add_keyframe_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client_id is not None
            camera = server.get_clients()[event.client_id].camera

            # Add this camera to the path.
            camera_path.add_camera(
                Keyframe.from_camera(
                    camera,
                    aspect=render_res_vec2.value[0] / render_res_vec2.value[1],
                ),
            )
            duration_number.value = camera_path.compute_duration()
            camera_path.update_spline()

        clear_keyframes_button = server.gui.add_button(
            "Clear Keyframes",
            icon=viser.Icon.TRASH,
            hint="Remove all keyframes from the render path.",
        )

        @clear_keyframes_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client_id is not None
            client = server.get_clients()[event.client_id]
            with client.atomic(), client.gui.add_modal("Confirm") as modal:
                client.gui.add_markdown("Clear all keyframes?")
                confirm_button = client.gui.add_button(
                    "Yes", color="red", icon=viser.Icon.TRASH
                )
                exit_button = client.gui.add_button("Cancel")

                @confirm_button.on_click
                def _(_) -> None:
                    camera_path.reset()
                    modal.close()

                    duration_number.value = camera_path.compute_duration()

                    # Clear move handles.
                    if len(transform_controls) > 0:
                        for t in transform_controls:
                            t.remove()
                        transform_controls.clear()
                        return

                @exit_button.on_click
                def _(_) -> None:
                    modal.close()

        reset_up_button = server.gui.add_button(
            "Reset Up Direction",
            icon=viser.Icon.ARROW_BIG_UP_LINES,
            color="gray",
            hint="Set the up direction of the camera orbit controls to the camera's current up direction.",
        )

        @reset_up_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            event.client.camera.up_direction = tf.SO3(
                event.client.camera.wxyz
            ) @ np.array([0.0, -1.0, 0.0])

        loop_checkbox = server.gui.add_checkbox(
            "Loop",
            False,
            hint="Add a segment between the first and last keyframes.",
        )

        @loop_checkbox.on_update
        def _(_) -> None:
            camera_path.loop = loop_checkbox.value
            duration_number.value = camera_path.compute_duration()

        tension_slider = server.gui.add_slider(
            "Spline tension",
            min=0.0,
            max=1.0,
            initial_value=0.0,
            step=0.01,
            hint="Tension parameter for adjusting smoothness of spline interpolation.",
        )

        @tension_slider.on_update
        def _(_) -> None:
            camera_path.tension = tension_slider.value
            camera_path.update_spline()

        move_checkbox = server.gui.add_checkbox(
            "Move keyframes",
            initial_value=False,
            hint="Toggle move handles for keyframes in the scene.",
        )

        transform_controls: List[viser.SceneNodeHandle] = []

        @move_checkbox.on_update
        def _(event: viser.GuiEvent) -> None:
            # Clear move handles when toggled off.
            if move_checkbox.value is False:
                for t in transform_controls:
                    t.remove()
                transform_controls.clear()
                return

            def _make_transform_controls_callback(
                keyframe: Tuple[Keyframe, viser.SceneNodeHandle],
                controls: viser.TransformControlsHandle,
            ) -> None:
                @controls.on_update
                def _(_) -> None:
                    keyframe[0].wxyz = controls.wxyz
                    keyframe[0].position = controls.position

                    keyframe[1].wxyz = controls.wxyz
                    keyframe[1].position = controls.position

                    camera_path.update_spline()

            # Show move handles.
            assert event.client is not None
            for keyframe_index, keyframe in camera_path._keyframes.items():
                controls = event.client.scene.add_transform_controls(
                    f"/keyframe_move/{keyframe_index}",
                    scale=0.4,
                    wxyz=keyframe[0].wxyz,
                    position=keyframe[0].position,
                )
                transform_controls.append(controls)
                _make_transform_controls_callback(keyframe, controls)

        show_keyframe_checkbox = server.gui.add_checkbox(
            "Show keyframes",
            initial_value=True,
            hint="Show keyframes in the scene.",
        )

        @show_keyframe_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            camera_path.set_keyframes_visible(show_keyframe_checkbox.value)

        show_spline_checkbox = server.gui.add_checkbox(
            "Show spline",
            initial_value=True,
            hint="Show camera path spline in the scene.",
        )

        @show_spline_checkbox.on_update
        def _(_) -> None:
            camera_path.show_spline = show_spline_checkbox.value
            camera_path.update_spline()

        transition_sec_number = server.gui.add_number(
            "Transition (sec)",
            min=0.001,
            max=30.0,
            step=0.001,
            initial_value=2.0,
            hint="Time in seconds between each keyframe, which can also be overridden on a per-transition basis.",
        )
        framerate_number = server.gui.add_number(
            "FPS", min=0.1, max=240.0, step=1e-2, initial_value=30.0
        )
        duration_number = server.gui.add_number(
            "Duration (sec)",
            min=0.0,
            max=1e8,
            step=0.001,
            initial_value=0.0,
            disabled=True,
        )

        @transition_sec_number.on_update
        def _(_) -> None:
            camera_path.default_transition_sec = transition_sec_number.value
            duration_number.value = camera_path.compute_duration()

        # set the initial value to the current date-time string
        trajectory_name_text = server.gui.add_text(
            "Name",
            initial_value="default",
            hint="Name of the trajectory",
        )

        # add button for loading existing path
        load_camera_path_button = server.gui.add_button(
            "Load Trajectory",
            icon=viser.Icon.FOLDER_OPEN,
            hint="Load an existing camera path.",
        )

        save_camera_path_button = server.gui.add_button(
            "Save Trajectory",
            icon=viser.Icon.FILE_EXPORT,
            hint="Save the current trajectory to a json file.",
        )

        play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        pause_button = server.gui.add_button(
            "Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False
        )
        preview_save_camera_path_button = server.gui.add_button(
            "Preview Render",
            icon=viser.Icon.EYE,
            hint="Show a preview of the render in the viewport.",
        )
        preview_render_stop_button = server.gui.add_button(
            "Exit Render Preview", color="red", visible=False
        )
        dump_video_button = server.gui.add_button(
            "Dump Video",
            color="green",
            icon=viser.Icon.PLAYER_PLAY,
            hint="Dump the current trajectory as a video.",
        )

    def get_max_frame_index() -> int:
        return max(1, int(framerate_number.value * duration_number.value) - 1)

    preview_camera_handle: Optional[viser.SceneNodeHandle] = None

    def remove_preview_camera() -> None:
        nonlocal preview_camera_handle
        if preview_camera_handle is not None:
            preview_camera_handle.remove()
            preview_camera_handle = None

    def compute_and_update_preview_camera_state() -> (
        Optional[Union[Tuple[tf.SE3, float], Tuple[tf.SE3, float, float]]]
    ):
        """Update the render tab state with the current preview camera pose.
        Returns current camera pose + FOV if available."""

        if preview_frame_slider is None:
            return
        maybe_pose_and_fov_rad = camera_path.interpolate_pose_and_fov_rad(
            preview_frame_slider.value / get_max_frame_index()
        )
        if maybe_pose_and_fov_rad is None:
            remove_preview_camera()
            return
        time = None
        if len(maybe_pose_and_fov_rad) == 3:  # Time is enabled.
            pose, fov_rad, time = maybe_pose_and_fov_rad
            render_tab_state.preview_time = time
        else:
            pose, fov_rad = maybe_pose_and_fov_rad
        render_tab_state.preview_fov = fov_rad
        render_tab_state.preview_aspect = camera_path.get_aspect()

        if time is not None:
            return pose, fov_rad, time
        else:
            return pose, fov_rad

    def add_preview_frame_slider() -> Optional[viser.GuiInputHandle[int]]:
        """Helper for creating the current frame # slider. This is removed and
        re-added anytime the `max` value changes."""

        with folder:
            preview_frame_slider = server.gui.add_slider(
                "Preview frame",
                min=0,
                max=get_max_frame_index(),
                step=1,
                initial_value=0,
                # Place right after the trajectory name text
                order=trajectory_name_text.order + 0.01,
                disabled=get_max_frame_index() == 1,
            )
            play_button.disabled = preview_frame_slider.disabled
            preview_save_camera_path_button.disabled = preview_frame_slider.disabled
            save_camera_path_button.disabled = preview_frame_slider.disabled
            dump_video_button.disabled = preview_frame_slider.disabled

        @preview_frame_slider.on_update
        def _(_) -> None:
            nonlocal preview_camera_handle
            maybe_pose_and_fov_rad = compute_and_update_preview_camera_state()
            if maybe_pose_and_fov_rad is None:
                return
            if len(maybe_pose_and_fov_rad) == 3:  # Time is enabled.
                pose, fov_rad, time = maybe_pose_and_fov_rad
            else:
                pose, fov_rad = maybe_pose_and_fov_rad

            preview_camera_handle = server.scene.add_camera_frustum(
                "/preview_camera",
                fov=fov_rad,
                aspect=render_res_vec2.value[0] / render_res_vec2.value[1],
                scale=0.35,
                wxyz=pose.rotation().wxyz,
                position=pose.translation(),
                color=(10, 200, 30),
            )
            if render_tab_state.preview_render:
                for client in server.get_clients().values():
                    # aspect ratio is not assignable, pass args in get_render instead
                    client.camera.wxyz = pose.rotation().wxyz
                    client.camera.position = pose.translation()
                    client.camera.fov = fov_rad

        return preview_frame_slider

    # We back up the camera poses before and after we start previewing renders.
    camera_pose_backup_from_id: Dict[int, tuple] = {}

    @preview_save_camera_path_button.on_click
    def _(_) -> None:
        render_tab_state.preview_render = True
        preview_save_camera_path_button.visible = False
        preview_render_stop_button.visible = True
        dump_video_button.disabled = True

        maybe_pose_and_fov_rad = compute_and_update_preview_camera_state()
        if maybe_pose_and_fov_rad is None:
            remove_preview_camera()
            return
        if len(maybe_pose_and_fov_rad) == 3:  # Time is enabled.
            pose, fov, time = maybe_pose_and_fov_rad
        else:
            pose, fov = maybe_pose_and_fov_rad
        del fov

        # Hide all scene nodes when we're previewing the render.
        server.scene.set_global_visibility(False)

        # Back up and then set camera poses.
        for client in server.get_clients().values():
            camera_pose_backup_from_id[client.client_id] = (
                client.camera.position,
                client.camera.look_at,
                client.camera.up_direction,
            )
            client.camera.wxyz = pose.rotation().wxyz
            client.camera.position = pose.translation()

    @preview_render_stop_button.on_click
    def _(_) -> None:
        render_tab_state.preview_render = False
        preview_save_camera_path_button.visible = True
        preview_render_stop_button.visible = False
        dump_video_button.disabled = False

        # Revert camera poses.
        for client in server.get_clients().values():
            if client.client_id not in camera_pose_backup_from_id:
                continue
            cam_position, cam_look_at, cam_up = camera_pose_backup_from_id.pop(
                client.client_id
            )
            client.camera.position = cam_position
            client.camera.look_at = cam_look_at
            client.camera.up_direction = cam_up
            client.flush()

        # Un-hide scene nodes.
        server.scene.set_global_visibility(True)

    preview_frame_slider = add_preview_frame_slider()
    handles = {
        "fov_degrees_slider": fov_degrees_slider,
        "render_res_vec2": render_res_vec2,
        "add_keyframe_button": add_keyframe_button,
        "clear_keyframes_button": clear_keyframes_button,
        "reset_up_button": reset_up_button,
        "loop_checkbox": loop_checkbox,
        "tension_slider": tension_slider,
        "move_checkbox": move_checkbox,
        "show_keyframe_checkbox": show_keyframe_checkbox,
        "show_spline_checkbox": show_spline_checkbox,
        "transition_sec_number": transition_sec_number,
        "framerate_number": framerate_number,
        "duration_number": duration_number,
        "trajectory_name_text": trajectory_name_text,
        "preview_frame_slider": preview_frame_slider,
        "load_camera_path_button": load_camera_path_button,
        "save_camera_path_button": save_camera_path_button,
        "play_button": play_button,
        "pause_button": pause_button,
        "preview_save_camera_path_button": preview_save_camera_path_button,
        "preview_render_stop_button": preview_render_stop_button,
        "dump_video_button": dump_video_button,
    }
    if time_enabled:
        handles["render_time"] = render_time

    # Update the # of frames.
    @duration_number.on_update
    @framerate_number.on_update
    def _(_) -> None:
        remove_preview_camera()  # Will be re-added when slider is updated.

        nonlocal preview_frame_slider
        old = preview_frame_slider
        assert old is not None

        preview_frame_slider = add_preview_frame_slider()
        if preview_frame_slider is not None:
            old.remove()
        else:
            preview_frame_slider = old

        handles["preview_frame_slider"] = preview_frame_slider
        camera_path.framerate = framerate_number.value
        camera_path.update_spline()

    # Play the camera trajectory when the play button is pressed.
    @play_button.on_click
    def _(_) -> None:
        play_button.visible = False
        pause_button.visible = True
        dump_video_button.disabled = True

        def play() -> None:
            while not play_button.visible:
                max_frame = int(framerate_number.value * duration_number.value)
                if max_frame > 0:
                    assert preview_frame_slider is not None
                    preview_frame_slider.value = (
                        preview_frame_slider.value + 1
                    ) % max_frame
                time.sleep(1.0 / framerate_number.value)

        play_thread = threading.Thread(target=play)
        play_thread.start()
        play_thread.join()
        dump_video_button.disabled = False

    # Play the camera trajectory when the play button is pressed.
    @pause_button.on_click
    def _(_) -> None:
        play_button.visible = True
        pause_button.visible = False

    @load_camera_path_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        camera_path_dir = output_dir / "camera_paths"
        camera_path_dir.mkdir(parents=True, exist_ok=True)
        preexisting_camera_paths = list(camera_path_dir.glob("*.json"))
        preexisting_camera_filenames = [p.name for p in preexisting_camera_paths]

        with event.client.gui.add_modal("Load Path") as modal:
            if len(preexisting_camera_filenames) == 0:
                event.client.gui.add_markdown("No existing paths found")
            else:
                event.client.gui.add_markdown("Select existing camera path:")
                camera_path_dropdown = event.client.gui.add_dropdown(
                    label="Camera Path",
                    options=[str(p) for p in preexisting_camera_filenames],
                    initial_value=str(preexisting_camera_filenames[0]),
                )
                load_button = event.client.gui.add_button("Load")

                @load_button.on_click
                def _(_) -> None:
                    # load the json file
                    json_path = output_dir / "camera_paths" / camera_path_dropdown.value
                    with open(json_path, "r") as f:
                        json_data = json.load(f)

                    keyframes = json_data["keyframes"]
                    camera_path.reset()
                    for i in range(len(keyframes)):
                        frame = keyframes[i]
                        pose = tf.SE3.from_matrix(
                            np.array(frame["matrix"]).reshape(4, 4)
                        )
                        # apply the x rotation by 180 deg
                        pose = tf.SE3.from_rotation_and_translation(
                            pose.rotation() @ tf.SO3.from_x_radians(np.pi),
                            pose.translation(),
                        )
                        camera_path.add_camera(
                            Keyframe(
                                position=pose.translation() * scale_ratio,
                                wxyz=pose.rotation().wxyz,
                                # There are some floating point conversions between degrees and radians, so the fov and
                                # default_Fov values will not be exactly matched.
                                override_fov_enabled=abs(
                                    frame["fov"] - json_data.get("default_fov", 0.0)
                                )
                                > 1e-3,
                                override_fov_rad=frame["fov"] / 180.0 * np.pi,
                                override_time_enabled=frame.get(
                                    "override_time_enabled", False
                                ),
                                override_time_val=frame.get("render_time", None),
                                aspect=frame["aspect"],
                                override_transition_enabled=frame.get(
                                    "override_transition_enabled", None
                                ),
                                override_transition_sec=frame.get(
                                    "override_transition_sec", None
                                ),
                            ),
                        )

                    transition_sec_number.value = json_data.get(
                        "default_transition_sec", 0.5
                    )

                    # update the render name
                    trajectory_name_text.value = json_path.stem
                    camera_path.update_spline()
                    modal.close()

                    # visualize the camera path
                    server.scene.set_global_visibility(True)

            cancel_button = event.client.gui.add_button("Cancel")

            @cancel_button.on_click
            def _(_) -> None:
                modal.close()

    @save_camera_path_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        num_frames = int(framerate_number.value * duration_number.value)
        json_data = {}
        # json data has the properties:
        # keyframes: list of keyframes with
        #     matrix : flattened 4x4 matrix
        #     fov: float in degrees
        #     aspect: float
        # render_height: int
        # render_width: int
        # fps: int
        # seconds: float
        # is_cycle: bool
        # smoothness_value: float
        # camera_path: list of frames with properties
        # camera_to_world: flattened 4x4 matrix
        # fov: float in degrees
        # aspect: float
        # first populate the keyframes:
        keyframes = []
        for keyframe, dummy in camera_path._keyframes.values():
            pose = tf.SE3.from_rotation_and_translation(
                tf.SO3(keyframe.wxyz) @ tf.SO3.from_x_radians(np.pi),
                keyframe.position / scale_ratio,
            )
            keyframe_dict = {
                "matrix": pose.as_matrix().flatten().tolist(),
                "fov": (
                    np.rad2deg(keyframe.override_fov_rad)
                    if keyframe.override_fov_enabled
                    else fov_degrees_slider.value
                ),
                "aspect": keyframe.aspect,
                "override_transition_enabled": keyframe.override_transition_enabled,
                "override_transition_sec": keyframe.override_transition_sec,
            }
            keyframes.append(keyframe_dict)
        json_data["default_fov"] = fov_degrees_slider.value
        json_data["default_transition_sec"] = transition_sec_number.value
        json_data["keyframes"] = keyframes
        json_data["render_height"] = render_res_vec2.value[1]
        json_data["render_width"] = render_res_vec2.value[0]
        json_data["fps"] = framerate_number.value
        json_data["seconds"] = duration_number.value
        json_data["is_cycle"] = loop_checkbox.value
        json_data["smoothness_value"] = tension_slider.value
        # now populate the camera path:
        camera_path_list = []
        for i in range(num_frames):
            maybe_pose_and_fov = camera_path.interpolate_pose_and_fov_rad(
                i / num_frames
            )
            if maybe_pose_and_fov is None:
                return
            time = None
            if len(maybe_pose_and_fov) == 3:  # Time is enabled.
                pose, fov, time = maybe_pose_and_fov
            else:
                pose, fov = maybe_pose_and_fov
            # rotate the axis of the camera 180 about x axis
            pose = tf.SE3.from_rotation_and_translation(
                pose.rotation() @ tf.SO3.from_x_radians(np.pi),
                pose.translation() / scale_ratio,
            )
            camera_path_list_dict = {
                "camera_to_world": pose.as_matrix().flatten().tolist(),
                "fov": np.rad2deg(fov),
                "aspect": render_res_vec2.value[0] / render_res_vec2.value[1],
            }
            if time is not None:
                camera_path_list_dict["render_time"] = time
            camera_path_list.append(camera_path_list_dict)
        json_data["camera_path"] = camera_path_list
        # finally add crop data if crop is enabled
        # if control_panel is not None:
        #     if control_panel.crop_viewport:
        #         obb = control_panel.crop_obb
        #         rpy = tf.SO3.from_matrix(obb.R.numpy()).as_rpy_radians()
        #         color = control_panel.background_color
        #         json_data["crop"] = {
        #             "crop_center": obb.T.tolist(),
        #             "crop_scale": obb.S.tolist(),
        #             "crop_rot": [rpy.roll, rpy.pitch, rpy.yaw],
        #             "crop_bg_color": {"r": color[0], "g": color[1], "b": color[2]},
        #         }

        # now write the json file
        try:
            json_outfile = (
                output_dir / "camera_paths" / f"{trajectory_name_text.value}.json"
            )
            json_outfile.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            Console(width=120).print(
                "[bold yellow]Warning: Failed to write the camera path to the data directory. Saving to the output directory instead."
            )
            json_outfile = (
                output_dir / "camera_paths" / f"{trajectory_name_text.value}.json"
            )
            json_outfile.parent.mkdir(parents=True, exist_ok=True)
        with open(json_outfile.absolute(), "w") as outfile:
            json.dump(json_data, outfile)
            print(f"Camera path saved to {json_outfile.absolute()}")

    @dump_video_button.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None

        # enter into preview render mode
        render_tab_state.preview_render = True
        maybe_pose_and_fov_rad = compute_and_update_preview_camera_state()
        if maybe_pose_and_fov_rad is None:
            remove_preview_camera()
            return
        if len(maybe_pose_and_fov_rad) == 3:  # Time is enabled.
            pose, fov, time = maybe_pose_and_fov_rad
        else:
            pose, fov = maybe_pose_and_fov_rad
        del fov

        # Hide all scene nodes when we're previewing the render.
        server.scene.set_global_visibility(False)

        # Back up and then set camera poses.
        for client in server.get_clients().values():
            camera_pose_backup_from_id[client.client_id] = (
                client.camera.position,
                client.camera.look_at,
                client.camera.up_direction,
            )
            client.camera.wxyz = pose.rotation().wxyz
            client.camera.position = pose.translation()

        # disable all the trajectory control widgets
        handles_to_disable = list(handles.values()) + list(extra_handles.values())
        original_disabled = [handle.disabled for handle in handles_to_disable]
        for handle in handles_to_disable:
            handle.disabled = True

        def dump() -> None:
            os.makedirs(output_dir / "videos", exist_ok=True)
            writer = imageio.get_writer(
                f"{output_dir}/videos/traj_{trajectory_name_text.value}.mp4",
                fps=framerate_number.value,
            )
            max_frame = int(framerate_number.value * duration_number.value)
            assert max_frame > 0 and preview_frame_slider is not None
            preview_frame_slider.value = 0
            for _ in range(max_frame):
                preview_frame_slider.value = (
                    preview_frame_slider.value + 1
                ) % max_frame
                # should we use get_render here?
                image = client.camera.get_render(
                    height=render_res_vec2.value[1],
                    width=render_res_vec2.value[0],
                )
                writer.append_data(image)
            writer.close()
            print(f"Video saved to videos/traj_{trajectory_name_text.value}.mp4")

        dump_thread = threading.Thread(target=dump)
        dump_thread.start()
        dump_thread.join()

        # restore the original disabled state
        for handle, original_disabled in zip(handles_to_disable, original_disabled):
            handle.disabled = original_disabled

        # exit preview render mode
        render_tab_state.preview_render = False

        # Revert camera poses.
        for client in server.get_clients().values():
            if client.client_id not in camera_pose_backup_from_id:
                continue
            cam_position, cam_look_at, cam_up = camera_pose_backup_from_id.pop(
                client.client_id
            )
            client.camera.position = cam_position
            client.camera.look_at = cam_look_at
            client.camera.up_direction = cam_up
            client.flush()

        # Un-hide scene nodes.
        server.scene.set_global_visibility(True)

    camera_path = CameraPath(server, duration_number)
    camera_path.tension = tension_slider.value
    camera_path.default_fov = fov_degrees_slider.value / 180.0 * np.pi
    camera_path.default_transition_sec = transition_sec_number.value

    return handles
