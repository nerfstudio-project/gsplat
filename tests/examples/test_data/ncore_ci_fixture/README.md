<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NCore AV trainer CI fixture

This fixture is derived from
[NCore clip `004c2001-5fc3-43b1-a4d8-bfb0bbb9fdc6`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore/tree/main/clips/004c2001-5fc3-43b1-a4d8-bfb0bbb9fdc6).
The clip metadata records source revision `ncore_test`.

It retains three original `camera_front_wide_120fov` frames, three matching
`lidar_top_360fov` spins, the original calibration and cuboids, 9,000 static
points, and 1,000 dynamic points owned by automobile track `13`. The reduced
sample is sufficient to exercise camera loading, static initialization,
dynamic-point removal, and rigid-track initialization without storing a full
NCore clip.

The `.zarr.itar` component stores are tracked with Git LFS. The JSON manifest
and component metadata also record the source clip, revision, selected frame
indices, track, and point counts.

## Fixture license and approval

Source: NVIDIA PhysicalAI Autonomous Vehicles NCore clip
`004c2001-5fc3-43b1-a4d8-bfb0bbb9fdc6`

Source license: NVIDIA Autonomous Vehicle Dataset License Agreement

Derivation: frames 123/126/129; LiDAR spins 40/41/42; 9,000 static + 1,000
track-13 points

Approved redistribution scope: Pending SWIPAT/OSRB approval - MR !378
