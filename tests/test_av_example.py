# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test for the autonomous driving training example.

Runs av_trainer.py with a small number of steps and verifies that the
training converges to a minimum PSNR threshold on held-out test views.
Requires CUDA and the test_pandaset.npz asset.
"""

import os
import sys
import tempfile

import pytest
import torch

import gsplat

SCRIPT_DIR = os.path.dirname(__file__)

sys.path.insert(0, os.path.join(SCRIPT_DIR, "../examples"))

# Import av_trainer behind importorskip so that this test module loads cleanly
# on environments (e.g. upstream GitHub Actions core_tests.yml on
# ubuntu-latest) that do not install av_trainer's optional dependencies
# (imageio, numpy stack beyond what gsplat itself needs, etc.). When any of
# those deps are missing the whole module is skipped at collection time
# instead of raising ModuleNotFoundError during import.
av_trainer = pytest.importorskip(
    "av_trainer",
    reason="av_trainer optional dependencies not installed (e.g. imageio)",
)

PANDASET_PATH = os.path.join(SCRIPT_DIR, "../assets/test_pandaset.npz")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support not built")
class TestAVExample:
    @pytest.fixture(autouse=True)
    def _check_asset(self):
        if not os.path.exists(PANDASET_PATH):
            pytest.skip(
                "PandaSet test asset not bundled in upstream repo. "
                f"Materialize {PANDASET_PATH} locally (e.g. via "
                "examples/prepare_pandaset.py against a user-provided "
                "PandaSet checkout) to run this test."
            )

    @pytest.mark.slow
    def test_training_converges(self):
        """Train for 15K steps and verify PSNR >= 16 dB on test views."""
        with tempfile.TemporaryDirectory() as tmpdir:
            losses, checkpoints = av_trainer.train(
                scene_path=PANDASET_PATH,
                max_steps=15000,
                lr=0.005,
                log_every=5000,
                eval_every=5000,
                result_dir=tmpdir,
            )

            # Loss must decrease
            assert (
                losses[-1] < losses[0]
            ), f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

            # Final PSNR must be at least 16 dB
            final_psnr = checkpoints[-1].get("mean_psnr", 0)
            assert (
                final_psnr >= 16.0
            ), f"Test PSNR too low: {final_psnr:.2f} dB (expected >= 16.0)"

    def test_quick_smoke(self):
        """Quick smoke test: 100 steps, just verify no crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            losses, checkpoints = av_trainer.train(
                scene_path=PANDASET_PATH,
                max_steps=100,
                lr=0.005,
                log_every=50,
                eval_every=100,
                result_dir=tmpdir,
            )

            assert len(losses) == 100
            assert all(loss > 0 for loss in losses)
            # Check eval ran and produced finite PSNR
            assert "mean_psnr" in checkpoints[-1]
            assert checkpoints[-1]["mean_psnr"] > 0
