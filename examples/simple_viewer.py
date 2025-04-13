"""A simple example to render a (large-scale) Gaussian Splats

```bash
python examples/simple_viewer.py --scene_grid 13
```
"""

import argparse
import math
import os
import time
from typing import Callable, Literal, Optional, Tuple, Union
from jaxtyping import Float32, UInt8

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser
import viser.transforms as vt

from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization

import dataclasses
from threading import Lock
from _renderer import Renderer, RenderTask


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
        mode: Literal["rendering", "training"] = "rendering",
    ):
        # Public states.
        self.server = server
        self.render_fn = render_fn
        self.mode = mode
        self.lock = VIEWER_LOCK
        self.state = ViewerState()
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
        with self.server.gui.add_folder(
            "Stats", visible=self.mode == "training"
        ) as self._stats_folder:
            self._stats_text_fn = (
                lambda: f"""<sub>
                Step: {self._step}\\
                Last Update: {self._last_update_step}
                </sub>"""
            )
            self._stats_text = self.server.gui.add_markdown(self._stats_text_fn())

        with self.server.gui.add_folder(
            "Training", visible=self.mode == "training"
        ) as self._training_folder:
            self._pause_train_button = self.server.gui.add_button("Pause")
            self._pause_train_button.on_click(self._toggle_train_buttons)
            self._pause_train_button.on_click(self._toggle_train_s)
            self._resume_train_button = self.server.gui.add_button("Resume")
            self._resume_train_button.visible = False
            self._resume_train_button.on_click(self._toggle_train_buttons)
            self._resume_train_button.on_click(self._toggle_train_s)

            self._train_util_slider = self.server.gui.add_slider(
                "Train Util", min=0.0, max=1.0, step=0.05, initial_value=0.9
            )
            self._train_util_slider.on_update(self.rerender)

        with self.server.gui.add_folder("Rendering") as self._rendering_folder:
            self._max_img_res_slider = self.server.gui.add_slider(
                "Max Img Res", min=64, max=2048, step=1, initial_value=2048
            )
            self._max_img_res_slider.on_update(self.rerender)

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
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=0 python -m simple_viewer \
        --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
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
