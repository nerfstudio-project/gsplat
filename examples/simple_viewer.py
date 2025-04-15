"""A simple example to render a (large-scale) Gaussian Splats

```bash
python examples/simple_viewer.py --scene_grid 13
```
"""

import argparse
import math
import os
import time
import colorsys
import json
import datetime
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union, Dict, List
from jaxtyping import Float32, UInt8
import dataclasses
import threading
from threading import Lock
from _renderer import Renderer, RenderTask
from scipy import interpolate
import splines
import splines.quaternion
from rich.console import Console
import imageio
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

import viser
import viser.transforms as vt
from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization


VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0
CONSOLE = Console(width=120)


@dataclasses.dataclass
class Keyframe:
    """
    Copied from nerfstudio/viewer/render_panel.py
    """

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
    def from_camera(camera: viser.CameraHandle, aspect: float) -> "Keyframe":
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
    """
    Copied from nerfstudio/viewer/render_panel.py
    """

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
                override_fov_degrees = server.gui.add_slider(
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
                override_fov_degrees.disabled = not override_fov.value
                self.add_camera(keyframe, keyframe_index)

            @override_fov_degrees.on_update
            def _(_) -> None:
                keyframe.override_fov_rad = override_fov_degrees.value / 180.0 * np.pi
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
                T_world_current = vt.SE3.from_rotation_and_translation(
                    vt.SO3(client.camera.wxyz), client.camera.position
                )
                T_world_target = vt.SE3.from_rotation_and_translation(
                    vt.SO3(keyframe.wxyz), keyframe.position
                ) @ vt.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(10):
                    T_world_set = T_world_current @ vt.SE3.exp(
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
    ) -> Optional[Union[Tuple[vt.SE3, float], Tuple[vt.SE3, float, float]]]:
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
                vt.SE3.from_rotation_and_translation(
                    vt.SO3(np.array([quat.scalar, *quat.vector])),
                    self._position_spline.evaluate(spline_t),
                ),
                float(self._fov_spline.evaluate(spline_t)),
                float(self._time_spline.evaluate(spline_t)),
            )
        else:
            return (
                vt.SE3.from_rotation_and_translation(
                    vt.SO3(np.array([quat.scalar, *quat.vector])),
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

    preview_render: bool
    preview_fov: float
    preview_time: float
    preview_aspect: float
    preview_camera_type: Literal["Perspective", "Fisheye", "Equirectangular"]


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: Float32[np.ndarray, "4 4"]

    def get_K(self, img_wh: Tuple[int, int]) -> Float32[np.ndarray, "3 3"]:
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K


@dataclasses.dataclass
class ViewerState(object):
    num_train_rays_per_sec: Optional[float] = None
    num_view_rays_per_sec: float = 100000.0
    status: Literal["rendering", "preparing", "training", "paused", "completed"] = (
        "training"
    )


VIEWER_LOCK = Lock()


def with_viewer_lock(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        with VIEWER_LOCK:
            return fn(*args, **kwargs)

    return wrapper


class Viewer(object):
    """Modified from nerfview.Viewer

    On instantiation, it (a) binds to a viser server and (b) creates a set of
    GUIs depending on its mode. After user connecting to the server, viewer
    renders and servers images in the background based on the camera movement.

    Args:
        server (viser.ViserServer): The viser server object to bind to.
        render_fn (Callable): A function that takes a camera state and image
            resolution as input and returns an image as a uint8 numpy array.
            Optionally, it can return a tuple of two images, where the second image
            is a float32 numpy depth map.
        mode (Literal["training", "rendering"]): The mode of the viewer.
            Support rendering and training. Defaults to "rendering".
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int]],
            Union[
                UInt8[np.ndarray, "H W 3"],
                Tuple[UInt8[np.ndarray, "H W 3"], Optional[Float32[np.ndarray, "H W"]]],
            ],
        ],
        datapath: Path,
        config_path: Path,
        mode: Literal["rendering", "training"] = "rendering",
    ):
        # Public states.
        self.server = server
        self.render_fn = render_fn
        self.mode = mode
        self.lock = VIEWER_LOCK
        self.state = ViewerState()
        self.datapath = datapath
        self.config_path = config_path
        if self.mode == "rendering":
            self.state.status = "rendering"

        # Private states.
        self._renderers: dict[int, Renderer] = {}
        self._step: int = 0
        self._last_update_step: int = 0
        self._last_move_time: float = 0.0

        server.on_client_disconnect(self._disconnect_client)
        server.on_client_connect(self._connect_client)

        self._define_guis()

    def _define_guis(self):
        server = self.server
        with server.gui.add_folder(
            "Stats", visible=self.mode == "training"
        ) as self._stats_folder:
            self._stats_text_fn = (
                lambda: f"""<sub>
                Step: {self._step}\\
                Last Update: {self._last_update_step}
                </sub>"""
            )
            self._stats_text = server.gui.add_markdown(self._stats_text_fn())

        with server.gui.add_folder(
            "Training", visible=self.mode == "training"
        ) as self._training_folder:
            self._pause_train_button = server.gui.add_button("Pause")
            self._pause_train_button.on_click(self._toggle_train_buttons)
            self._pause_train_button.on_click(self._toggle_train_s)
            self._resume_train_button = server.gui.add_button("Resume")
            self._resume_train_button.visible = False
            self._resume_train_button.on_click(self._toggle_train_buttons)
            self._resume_train_button.on_click(self._toggle_train_s)

            self._train_util_slider = server.gui.add_slider(
                "Train Util", min=0.0, max=1.0, step=0.05, initial_value=0.9
            )
            self._train_util_slider.on_update(self.rerender)

        with server.gui.add_folder("Rendering") as self._rendering_folder:
            self._max_img_res_slider = server.gui.add_slider(
                "Max Img Res", min=64, max=2048, step=1, initial_value=2048
            )
            self._max_img_res_slider.on_update(self.rerender)

            render_tab_state = RenderTabState(
                preview_render=False,
                preview_fov=0.0,
                preview_time=0.0,
                preview_aspect=1.0,
                preview_camera_type="Perspective",
            )

            fov_degrees = server.gui.add_slider(
                "Default FOV",
                initial_value=75.0,
                min=0.1,
                max=175.0,
                step=0.01,
                hint="Field-of-view for rendering, which can also be overridden on a per-keyframe basis.",
            )

            @fov_degrees.on_update
            def _(_) -> None:
                fov_radians = fov_degrees.value / 180.0 * np.pi
                for client in server.get_clients().values():
                    client.camera.fov = fov_radians

                camera_path.default_fov = fov_radians

                # Updating the aspect ratio will also re-render the camera frustums.
                # Could rethink this.
                camera_path.update_aspect(resolution.value[0] / resolution.value[1])
                compute_and_update_preview_camera_state()

            resolution = server.gui.add_vector2(
                "Resolution",
                initial_value=(1920, 1080),
                min=(50, 50),
                max=(10_000, 10_000),
                step=1,
                hint="Render output resolution in pixels.",
            )

            @resolution.on_update
            def _(_) -> None:
                camera_path.update_aspect(resolution.value[0] / resolution.value[1])
                compute_and_update_preview_camera_state()

            camera_type = server.gui.add_dropdown(
                "Camera type",
                ("Perspective", "Fisheye", "Equirectangular"),
                initial_value="Perspective",
                hint="Camera model to render with. This is applied to all keyframes.",
            )
            add_button = server.gui.add_button(
                "Add Keyframe",
                icon=viser.Icon.PLUS,
                hint="Add a new keyframe at the current pose.",
            )

            @add_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client_id is not None
                camera = server.get_clients()[event.client_id].camera

                # Add this camera to the path.
                camera_path.add_camera(
                    Keyframe.from_camera(
                        camera,
                        aspect=resolution.value[0] / resolution.value[1],
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

            loop = server.gui.add_checkbox(
                "Loop",
                False,
                hint="Add a segment between the first and last keyframes.",
            )

            @loop.on_update
            def _(_) -> None:
                camera_path.loop = loop.value
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

            playback_folder = server.gui.add_folder("Playback")
            with playback_folder:
                play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
                pause_button = server.gui.add_button(
                    "Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False
                )
                preview_render_button = server.gui.add_button(
                    "Preview Render",
                    hint="Show a preview of the render in the viewport.",
                )
                preview_render_stop_button = server.gui.add_button(
                    "Exit Render Preview", color="red", visible=False
                )

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
            framerate_buttons = server.gui.add_button_group("", ("24", "30", "60"))
            duration_number = server.gui.add_number(
                "Duration (sec)",
                min=0.0,
                max=1e8,
                step=0.001,
                initial_value=0.0,
                disabled=True,
            )

            @framerate_buttons.on_click
            def _(_) -> None:
                framerate_number.value = float(framerate_buttons.value)

            @transition_sec_number.on_update
            def _(_) -> None:
                camera_path.default_transition_sec = transition_sec_number.value
                duration_number.value = camera_path.compute_duration()

        def get_max_frame_index() -> int:
            return max(1, int(framerate_number.value * duration_number.value) - 1)

        preview_camera_handle: Optional[viser.SceneNodeHandle] = None

        def remove_preview_camera() -> None:
            nonlocal preview_camera_handle
            if preview_camera_handle is not None:
                preview_camera_handle.remove()
                preview_camera_handle = None

        def compute_and_update_preview_camera_state() -> (
            Optional[Union[Tuple[vt.SE3, float], Tuple[vt.SE3, float, float]]]
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
            render_tab_state.preview_camera_type = camera_type.value

            if time is not None:
                return pose, fov_rad, time
            else:
                return pose, fov_rad

        def add_preview_frame_slider() -> Optional[viser.GuiInputHandle[int]]:
            """Helper for creating the current frame # slider. This is removed and
            re-added anytime the `max` value changes."""

            with playback_folder:
                preview_frame_slider = server.gui.add_slider(
                    "Preview frame",
                    min=0,
                    max=get_max_frame_index(),
                    step=1,
                    initial_value=0,
                    # Place right after the pause button.
                    order=preview_render_stop_button.order + 0.01,
                    disabled=get_max_frame_index() == 1,
                )
                play_button.disabled = preview_frame_slider.disabled
                preview_render_button.disabled = preview_frame_slider.disabled

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
                    aspect=resolution.value[0] / resolution.value[1],
                    scale=0.35,
                    wxyz=pose.rotation().wxyz,
                    position=pose.translation(),
                    color=(10, 200, 30),
                )
                if render_tab_state.preview_render:
                    for client in server.get_clients().values():
                        client.camera.wxyz = pose.rotation().wxyz
                        client.camera.position = pose.translation()

            return preview_frame_slider

        # We back up the camera poses before and after we start previewing renders.
        camera_pose_backup_from_id: Dict[int, tuple] = {}

        @preview_render_button.on_click
        def _(_) -> None:
            render_tab_state.preview_render = True
            preview_render_button.visible = False
            preview_render_stop_button.visible = True

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
            preview_render_button.visible = True
            preview_render_stop_button.visible = False

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

            camera_path.framerate = framerate_number.value
            camera_path.update_spline()

        # Play the camera trajectory when the play button is pressed.
        @play_button.on_click
        def _(_) -> None:
            play_button.visible = False
            pause_button.visible = True

            def play() -> None:
                while not play_button.visible:
                    max_frame = int(framerate_number.value * duration_number.value)
                    if max_frame > 0:
                        assert preview_frame_slider is not None
                        preview_frame_slider.value = (
                            preview_frame_slider.value + 1
                        ) % max_frame
                    time.sleep(1.0 / framerate_number.value)

            threading.Thread(target=play).start()

        # Play the camera trajectory when the play button is pressed.
        @pause_button.on_click
        def _(_) -> None:
            play_button.visible = True
            pause_button.visible = False

        # add button for loading existing path
        load_camera_path_button = server.gui.add_button(
            "Load Path",
            icon=viser.Icon.FOLDER_OPEN,
            hint="Load an existing camera path.",
        )

        @load_camera_path_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            camera_path_dir = self.datapath / "camera_paths"
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
                        json_path = (
                            self.datapath / "camera_paths" / camera_path_dropdown.value
                        )
                        with open(json_path, "r") as f:
                            json_data = json.load(f)

                        keyframes = json_data["keyframes"]
                        camera_path.reset()
                        for i in range(len(keyframes)):
                            frame = keyframes[i]
                            pose = vt.SE3.from_matrix(
                                np.array(frame["matrix"]).reshape(4, 4)
                            )
                            # apply the x rotation by 180 deg
                            pose = vt.SE3.from_rotation_and_translation(
                                pose.rotation() @ vt.SO3.from_x_radians(np.pi),
                                pose.translation(),
                            )
                            camera_path.add_camera(
                                Keyframe(
                                    position=pose.translation()
                                    * VISER_NERFSTUDIO_SCALE_RATIO,
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
                        render_name_text.value = json_path.stem
                        camera_path.update_spline()
                        modal.close()

                cancel_button = event.client.gui.add_button("Cancel")

                @cancel_button.on_click
                def _(_) -> None:
                    modal.close()

        # set the initial value to the current date-time string
        # now = datetime.datetime.now()
        render_name_text = server.gui.add_text(
            "Render name",
            initial_value="default",
            hint="Name of the render",
        )
        render_button = server.gui.add_button(
            "Generate Command",
            color="green",
            icon=viser.Icon.FILE_EXPORT,
            hint="Generate the ns-render command for rendering the camera path.",
        )

        reset_up_button = server.gui.add_button(
            "Reset Up Direction",
            icon=viser.Icon.ARROW_BIG_UP_LINES,
            color="gray",
            hint="Set the up direction of the camera orbit controls to the camera's current up direction.",
        )

        @reset_up_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            event.client.camera.up_direction = vt.SO3(
                event.client.camera.wxyz
            ) @ np.array([0.0, -1.0, 0.0])

        @render_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            num_frames = int(framerate_number.value * duration_number.value)
            json_data = {}
            # json data has the properties:
            # keyframes: list of keyframes with
            #     matrix : flattened 4x4 matrix
            #     fov: float in degrees
            #     aspect: float
            # camera_type: string of camera type
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
                pose = vt.SE3.from_rotation_and_translation(
                    vt.SO3(keyframe.wxyz) @ vt.SO3.from_x_radians(np.pi),
                    keyframe.position / VISER_NERFSTUDIO_SCALE_RATIO,
                )
                keyframe_dict = {
                    "matrix": pose.as_matrix().flatten().tolist(),
                    "fov": (
                        np.rad2deg(keyframe.override_fov_rad)
                        if keyframe.override_fov_enabled
                        else fov_degrees.value
                    ),
                    "aspect": keyframe.aspect,
                    "override_transition_enabled": keyframe.override_transition_enabled,
                    "override_transition_sec": keyframe.override_transition_sec,
                }
                keyframes.append(keyframe_dict)
            json_data["default_fov"] = fov_degrees.value
            json_data["default_transition_sec"] = transition_sec_number.value
            json_data["keyframes"] = keyframes
            json_data["camera_type"] = camera_type.value.lower()
            json_data["render_height"] = resolution.value[1]
            json_data["render_width"] = resolution.value[0]
            json_data["fps"] = framerate_number.value
            json_data["seconds"] = duration_number.value
            json_data["is_cycle"] = loop.value
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
                pose = vt.SE3.from_rotation_and_translation(
                    pose.rotation() @ vt.SO3.from_x_radians(np.pi),
                    pose.translation() / VISER_NERFSTUDIO_SCALE_RATIO,
                )
                camera_path_list_dict = {
                    "camera_to_world": pose.as_matrix().flatten().tolist(),
                    "fov": np.rad2deg(fov),
                    "aspect": resolution.value[0] / resolution.value[1],
                }
                if time is not None:
                    camera_path_list_dict["render_time"] = time
                camera_path_list.append(camera_path_list_dict)
            json_data["camera_path"] = camera_path_list
            # finally add crop data if crop is enabled
            # if control_panel is not None:
            #     if control_panel.crop_viewport:
            #         obb = control_panel.crop_obb
            #         rpy = vt.SO3.from_matrix(obb.R.numpy()).as_rpy_radians()
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
                    self.datapath / "camera_paths" / f"{render_name_text.value}.json"
                )
                json_outfile.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                CONSOLE.print(
                    "[bold yellow]Warning: Failed to write the camera path to the data directory. Saving to the output directory instead."
                )
                json_outfile = (
                    self.config_path.parent
                    / "camera_paths"
                    / f"{render_name_text.value}.json"
                )
                json_outfile.parent.mkdir(parents=True, exist_ok=True)
            with open(json_outfile.absolute(), "w") as outfile:
                json.dump(json_data, outfile)

            # instead of showing the command, directly render the video

            # now show the command
            # with event.client.gui.add_modal("Render Command") as modal:
            #     dataname = self.datapath.name
            #     command = " ".join(
            #         [
            #             "ns-render camera-path",
            #             f"--load-config {self.config_path}",
            #             f"--camera-path-filename {json_outfile.absolute()}",
            #             f"--output-path renders/{dataname}/{render_name_text.value}.mp4",
            #         ]
            #     )
            #     event.client.gui.add_markdown(
            #         "\n".join(
            #             [
            #                 "To render the trajectory, run the following from the command line:",
            #                 "",
            #                 "```",
            #                 command,
            #                 "```",
            #             ]
            #         )
            #     )
            #     close_button = event.client.gui.add_button("Close")

            #     @close_button.on_click
            #     def _(_) -> None:
            #         modal.close()

        camera_path = CameraPath(server, duration_number)
        camera_path.tension = tension_slider.value
        camera_path.default_fov = fov_degrees.value / 180.0 * np.pi
        camera_path.default_transition_sec = transition_sec_number.value

    def _toggle_train_buttons(self, _):
        self._pause_train_button.visible = not self._pause_train_button.visible
        self._resume_train_button.visible = not self._resume_train_button.visible

    def _toggle_train_s(self, _):
        if self.state.status == "completed":
            return
        self.state.status = "paused" if self.state.status == "training" else "training"

    def rerender(self, _):
        clients = self.server.get_clients()
        for client_id in clients:
            camera_state = self.get_camera_state(clients[client_id])
            assert camera_state is not None
            self._renderers[client_id].submit(RenderTask("rerender", camera_state))

    def _disconnect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id].running = False
        self._renderers.pop(client_id)

    def _connect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id] = Renderer(
            viewer=self, client=client, lock=self.lock
        )
        self._renderers[client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._last_move_time = time.time()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [vt.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    def update(self, step: int, num_train_rays_per_step: int):
        if self.mode == "rendering":
            raise ValueError("`update` method is only available in training mode.")
        # Skip updating the viewer for the first few steps to allow
        # `num_train_rays_per_sec` and `num_view_rays_per_sec` to stabilize.
        if step < 5:
            return
        self._step = step
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = self._stats_text_fn()
        if len(self._renderers) == 0:
            return
        # Stop training while user moves camera to make viewing smoother.
        while time.time() - self._last_move_time < 0.1:
            time.sleep(0.05)
        if self.state.status == "training" and self._train_util_slider.value != 1:
            assert (
                self.state.num_train_rays_per_sec is not None
            ), "User must keep track of `num_train_rays_per_sec` to use `update`."
            train_s = self.state.num_train_rays_per_sec
            view_s = self.state.num_view_rays_per_sec
            train_util = self._train_util_slider.value
            view_n = self._max_img_res_slider.value**2
            train_n = num_train_rays_per_step
            train_time = train_n / train_s
            view_time = view_n / view_s
            update_every = (
                train_util * view_time / (train_time - train_util * train_time)
            )
            if step > self._last_update_step + update_every:
                self._last_update_step = step
                clients = self.server.get_clients()
                for client_id in clients:
                    camera_state = self.get_camera_state(clients[client_id])
                    assert camera_state is not None
                    self._renderers[client_id].submit(
                        RenderTask("update", camera_state)
                    )
                with self.server.atomic(), self._stats_folder:
                    self._stats_text.content = self._stats_text_fn()

    def complete(self):
        self.state.status = "completed"
        self._pause_train_button.disabled = True
        self._resume_train_button.disabled = True
        self._train_util_slider.disabled = True
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = f"""<sub>
                Step: {self._step}\\
                Training Completed!
                </sub>"""


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.ckpt is None:
        (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
        ) = load_test_data(device=device, scene_grid=args.scene_grid)

        assert world_size <= 2
        means = means[world_rank::world_size].contiguous()
        means.requires_grad = True
        quats = quats[world_rank::world_size].contiguous()
        quats.requires_grad = True
        scales = scales[world_rank::world_size].contiguous()
        scales.requires_grad = True
        opacities = opacities[world_rank::world_size].contiguous()
        opacities.requires_grad = True
        colors = colors[world_rank::world_size].contiguous()
        colors.requires_grad = True

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()

        sh_degree = None
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)

        # batched render
        for _ in tqdm.trange(1):
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                width,
                height,
                render_mode="RGB+D",
                packed=False,
                distributed=world_size > 1,
            )
        C = render_colors.shape[0]
        assert render_colors.shape == (C, height, width, 4)
        assert render_alphas.shape == (C, height, width, 1)
        render_colors.sum().backward()

        render_rgbs = render_colors[..., 0:3]
        render_depths = render_colors[..., 3:4]
        render_depths = render_depths / render_depths.max()

        # dump batch images
        os.makedirs(args.output_dir, exist_ok=True)
        canvas = (
            torch.cat(
                [
                    render_rgbs.reshape(C * height, width, 3),
                    render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                    render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        imageio.imsave(
            f"{args.output_dir}/render_rank{world_rank}.png",
            (canvas * 255).astype(np.uint8),
        )
    else:
        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]
            means.append(ckpt["means"])
            quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
            scales.append(torch.exp(ckpt["scales"]))
            opacities.append(torch.sigmoid(ckpt["opacities"]))
            sh0.append(ckpt["sh0"])
            shN.append(ckpt["shN"])
        means = torch.cat(means, dim=0)
        quats = torch.cat(quats, dim=0)
        scales = torch.cat(scales, dim=0)
        opacities = torch.cat(opacities, dim=0)
        sh0 = torch.cat(sh0, dim=0)
        shN = torch.cat(shN, dim=0)
        colors = torch.cat([sh0, shN], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

        # # crop
        # aabb = torch.tensor((-1.0, -1.0, -1.0, 1.0, 1.0, 0.7), device=device)
        # edges = aabb[3:] - aabb[:3]
        # sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
        # sel = torch.where(sel)[0]
        # means, quats, scales, colors, opacities = (
        #     means[sel],
        #     quats[sel],
        #     scales[sel],
        #     colors[sel],
        #     opacities[sel],
        # )

        # # repeat the scene into a grid (to mimic a large-scale setting)
        # repeats = args.scene_grid
        # gridx, gridy = torch.meshgrid(
        #     [
        #         torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        #         torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        #     ],
        #     indexing="ij",
        # )
        # grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(
        #     -1, 3
        # )
        # means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
        # means = means.reshape(-1, 3)
        # quats = quats.repeat(repeats**2, 1)
        # scales = scales.repeat(repeats**2, 1)
        # colors = colors.repeat(repeats**2, 1, 1)
        # opacities = opacities.repeat(repeats**2)
        print("Number of Gaussians:", len(means))

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        if args.backend == "gsplat":
            rasterization_fn = rasterization
        elif args.backend == "inria":
            from gsplat import rasterization_inria_wrapper

            rasterization_fn = rasterization_inria_wrapper
        else:
            raise ValueError

        render_colors, render_alphas, meta = rasterization_fn(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=sh_degree,
            render_mode="RGB",
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            radius_clip=3,
        )
        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = Viewer(
        server=server,
        render_fn=viewer_render_fn,
        datapath=Path(args.output_dir),
        config_path=Path(args.output_dir),
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
        --output_dir results/garden/ \
        --port 8081
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument("--backend", type=str, default="gsplat", help="gsplat, inria")
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
