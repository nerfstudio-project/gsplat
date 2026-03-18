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

ALPHA_THRESHOLD = 1.0 / 255.0
# MAX_ALPHA and TRANSMITTANCE_THRESHOLD are chosen so that the equivalent of
# a maximal opacity Gaussian has to be rasterized twice to reach the threshold,
# without getting the transmittance too small for numerical stability of
# the backward pass.
# i.e. TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2
MAX_ALPHA = 0.99
TRANSMITTANCE_THRESHOLD = 1e-4

MAX_KERNEL_DENSITY_CUTOFF = 0.0113
