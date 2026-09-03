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

from ._rendering import NHTParams
from ._inference_renderer import NHTInferenceConfig, NHTInferenceRenderer
from ._wrapper import convert_mlp_params_to_fused_native
from ._fused_train import nht_fused_render, nht_fused_supported
from .deferred_shader import (
    DeferredShaderModule,
    DeferredShaderModuleAOV,
    HarmonicFeatures,
)
from .exporter import (
    cast_state_dict_to_fp16,
    cast_state_dict_to_fp32,
    export_splats_nht,
)

__all__ = [
    "DeferredShaderModule",
    "DeferredShaderModuleAOV",
    "HarmonicFeatures",
    "NHTInferenceConfig",
    "NHTInferenceRenderer",
    "NHTParams",
    "cast_state_dict_to_fp16",
    "cast_state_dict_to_fp32",
    "convert_mlp_params_to_fused_native",
    "export_splats_nht",
    "nht_fused_render",
    "nht_fused_supported",
]
