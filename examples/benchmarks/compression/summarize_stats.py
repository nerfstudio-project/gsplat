# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import json
import os
import subprocess
from collections import defaultdict
from typing import List

import numpy as np
import tyro


def main(results_dir: str, scenes: List[str], stage: str = "compress"):
    print("scenes:", scenes)

    summary = defaultdict(list)
    for scene in scenes:
        scene_dir = os.path.join(results_dir, scene)

        if stage == "compress":
            zip_path = f"{scene_dir}/compression.zip"
            if os.path.exists(zip_path):
                subprocess.run(f"rm {zip_path}", shell=True)
            subprocess.run(f"zip -r {zip_path} {scene_dir}/compression/", shell=True)
            out = subprocess.run(
                f"stat -c%s {zip_path}", shell=True, capture_output=True
            )
            size = int(out.stdout)
            summary["size"].append(size)

        with open(os.path.join(scene_dir, f"stats/{stage}_step29999.json"), "r") as f:
            stats = json.load(f)
            for k, v in stats.items():
                summary[k].append(v)

    for k, v in summary.items():
        summary[k] = np.mean(v)
    summary["scenes"] = scenes

    with open(os.path.join(results_dir, f"{stage}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    tyro.cli(main)
