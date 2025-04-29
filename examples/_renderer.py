"""
Modified from nerfview/_renderer.py
"""

import dataclasses
import os
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING, Literal, Optional, Tuple, get_args

import viser

if TYPE_CHECKING:
    from examples.viewer import CameraState, Viewer

RenderState = Literal["low_move", "low_static", "high"]
RenderAction = Literal["rerender", "move", "static", "update"]


@dataclasses.dataclass
class RenderTask(object):
    action: RenderAction
    camera_state: Optional["CameraState"] = None


class InterruptRenderException(Exception):
    pass


class set_trace_context(object):
    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, *_, **__):
        sys.settrace(None)


class Renderer(threading.Thread):
    """This class is responsible for rendering images in the background."""

    def __init__(
        self,
        viewer: "Viewer",
        client: viser.ClientHandle,
        lock: threading.Lock,
    ):
        super().__init__(daemon=True)

        self.viewer = viewer
        self.client = client
        self.lock = lock

        self.running = True
        self.is_prepared_fn = lambda: self.viewer.state != "preparing"

        self._render_event = threading.Event()
        self._state: RenderState = "low_static"
        self._task: Optional[RenderTask] = None

        self._target_fps = 30
        self._may_interrupt_render = False
        self._old_version = False

        self._define_transitions()

    def _define_transitions(self):
        transitions: dict[RenderState, dict[RenderAction, RenderState]] = {
            s: {a: s for a in get_args(RenderAction)} for s in get_args(RenderState)
        }
        transitions["low_move"]["static"] = "low_static"
        transitions["low_static"]["static"] = "high"
        transitions["low_static"]["update"] = "high"
        transitions["low_static"]["move"] = "low_move"
        transitions["high"]["move"] = "low_move"
        transitions["high"]["rerender"] = "low_static"
        self.transitions = transitions

    def _may_interrupt_trace(self, frame, event, arg):
        if event == "line":
            if self._may_interrupt_render:
                self._may_interrupt_render = False
                raise InterruptRenderException
        return self._may_interrupt_trace

    def _get_img_wh(self, aspect: float) -> Tuple[int, int]:
        # we always trade off speed for quality
        max_img_res = self.viewer.render_tab_state.viewer_res
        if self._state in ["high"]:
            #  if True:
            H = max_img_res
            W = int(H * aspect)
            if W > max_img_res:
                W = max_img_res
                H = int(W / aspect)
        elif self._state in ["low_move", "low_static"]:
            num_view_rays_per_sec = self.viewer.render_tab_state.num_view_rays_per_sec
            target_fps = self._target_fps
            num_viewer_rays = num_view_rays_per_sec / target_fps
            H = (num_viewer_rays / aspect) ** 0.5
            H = int(round(H, -1))
            H = max(min(max_img_res, H), 30)
            W = int(H * aspect)
            if W > max_img_res:
                W = max_img_res
                H = int(W / aspect)
        else:
            raise ValueError(f"Unknown state: {self._state}.")
        return W, H

    def submit(self, task: RenderTask):
        if self._task is None:
            self._task = task
        elif task.action == "update" and (
            self._state == "low_move" or self._task.action in ["move", "rerender"]
        ):
            return
        else:
            self._task = task

        if self._state == "high" and self._task.action in ["move", "rerender"]:
            self._may_interrupt_render = True
        self._render_event.set()

    def run(self):
        while self.running:
            while not self.is_prepared_fn():
                time.sleep(0.1)
            if not self._render_event.wait(0.2):
                self.submit(
                    RenderTask("static", self.viewer.get_camera_state(self.client))
                )
            self._render_event.clear()
            task = self._task
            assert task is not None
            #  print(self._state, task.action, self.transitions[self._state][task.action])
            if self._state == "high" and task.action == "static":
                continue
            self._state = self.transitions[self._state][task.action]
            assert task.camera_state is not None
            try:
                with self.lock, set_trace_context(self._may_interrupt_trace):
                    tic = time.time()
                    W, H = img_wh = self._get_img_wh(task.camera_state.aspect)
                    self.viewer.render_tab_state.viewer_width = W
                    self.viewer.render_tab_state.viewer_height = H

                    if not self._old_version:
                        try:
                            rendered = self.viewer.render_fn(
                                task.camera_state,
                                self.viewer.render_tab_state,
                            )
                        except TypeError:
                            self._old_version = True
                            print(
                                "[WARNING] Your API will be deprecated in the future, please update your render_fn."
                            )
                            rendered = self.viewer.render_fn(task.camera_state, img_wh)
                    else:
                        rendered = self.viewer.render_fn(task.camera_state, img_wh)

                    self.viewer._after_render()
                    if isinstance(rendered, tuple):
                        img, depth = rendered
                    else:
                        img, depth = rendered, None
                    self.viewer.render_tab_state.num_view_rays_per_sec = (W * H) / (
                        time.time() - tic
                    )
            except InterruptRenderException:
                continue
            except Exception:
                traceback.print_exc()
                os._exit(1)
            self.client.scene.set_background_image(
                img,
                format="jpeg",
                jpeg_quality=70 if task.action in ["static", "update"] else 40,
                depth=depth,
            )
