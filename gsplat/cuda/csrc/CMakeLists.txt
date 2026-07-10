# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# Internal CUDA implementation
# -----------------------------------------------------------------------------

include(GSplatTargetSourceInstances)

# New implementation files must be added explicitly to the source lists on
# the target below; build sources are not globbed.
add_library(
    gsplat_core
    SHARED
    Adam.cpp
    BgGridLosses.cpp
    BgTrackNodeSemanticLosses.cpp
    CameraLosses.cpp
    DeformLosses.cpp
    DistributedCollectives.cpp
    GaussianLosses.cpp
    GroundGaussiansLosses.cpp
    Intersect.cpp
    IntersectMacroTile.cu
    IntersectTile.cu
    IntersectTileLidar.cu
    IntersectTileSparse.cu
    Lidar.cpp
    LidarLosses.cpp
    MCMCPerturb.cpp
    Null.cpp
    NullCUDA.cu
    Projection.cpp
    QuatScaleToCovar.cpp
    Rasterization.cpp
    Relocation.cpp
    Rendering.cpp
    SegmentedSort.cu
    SemanticCELosses.cpp
    SparseTileLayout.cu
    SphericalHarmonics.cpp
    SphericalHarmonicsCUDA.cu
    SphericalHarmonicsL1PlusCUDA.cu
    SsimLosses.cpp
    Utils.cpp
)

set(_gsplat_instance_sources)
if(_gsplat_final_build_2DGS)
    target_sources(
        gsplat_core
        PRIVATE
            Projection2DGSFused.cu
            Projection2DGSPacked.cu
            RasterizeToIndices2DGSSerialBatch.cu
            RasterizeToPixels2DGSSerialBatchBwd.cu
            RasterizeToPixels2DGSSerialBatchFwd.cu
    )
endif()
if(_gsplat_final_build_3DGS)
    target_sources(
        gsplat_core
        PRIVATE
            MCMCPerturbCUDA.cu
            ProjectionEWA3DGSFused.cu
            ProjectionEWA3DGSPacked.cu
            ProjectionEWASimple.cu
            QuatScaleToCovarCUDA.cu
            RasterizeContributingGaussianIds.cu
            RasterizeNumContributingGaussians.cu
            RasterizeToIndices3DGSSerialBatch.cu
            RasterizeToPixels3DGSSerialBatchBwd.cu
            RasterizeToPixels3DGSSerialBatchFwd.cu
            RasterizeToPixelsSparseBwd.cu
            RasterizeToPixelsSparseFwd.cu
            RasterizeTopContributingGaussianIds.cu
    )
endif()
if(_gsplat_final_build_3DGUT)
    target_sources(
        gsplat_core
        PRIVATE
            PrimingChainEncoding.cu
            ProjectionUT3DGSFused.cu
            RasterizeToPixelsFromWorld3DGSParallelBatchBwd.cu
    )
    # These translation units have enough channel-dependent compiler work to
    # benefit from separate channel instantiations. Keep the longest source
    # first so parallel build tools can begin it as early as possible.
    list(
        APPEND _gsplat_instance_sources
        RasterizeToPixelsFromWorld3DGSParallelBatchFwd.cu
        RasterizeToPixelsFromWorld3DGSSerialBatchFwd.cu
    )
endif()
if(_gsplat_final_build_ADAM)
    target_sources(gsplat_core PRIVATE AdamCUDA.cu)
endif()
if(_gsplat_final_build_RELOC)
    target_sources(gsplat_core PRIVATE RelocationCUDA.cu)
endif()
if(_gsplat_final_build_LOSSES)
    target_sources(
        gsplat_core
        PRIVATE
            BgGridLossesCUDA.cu
            BgTrackNodeSemanticLossesCUDA.cu
            CameraLossesCUDA.cu
            DeformLossesCUDA.cu
            GaussianLossesCUDA.cu
            GroundGaussiansLossesCUDA.cu
            LidarLossesCUDA.cu
            SemanticCELossesCUDA.cu
            SsimLossesCUDA.cu
    )
endif()
if(_gsplat_final_build_camera_wrappers)
    target_sources(gsplat_core PRIVATE CameraWrappers.cu ExternalDistortionWrappers.cu)
endif()

# Instance sources also need one compilation without
# GSPLAT_INSTANTIATE_TEMPLATE to provide their runtime dispatch functions.
target_sources(gsplat_core PRIVATE ${_gsplat_instance_sources})
set_target_properties(gsplat_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(
    gsplat_core
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${_gsplat_python_package_dir}/lib"
        BUILD_RPATH_USE_ORIGIN ON
)
target_include_directories(
    gsplat_core
    BEFORE
    PUBLIC
        "${GSPLAT_GENERATED_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(
    gsplat_core
    PUBLIC gsplat_compile_opts gsplat::ext::glm gsplat::ext::torch CUDA::cudart
    PRIVATE Python::Module
)

# Larger channel counts take longer to compile, so schedule them first without
# changing the configured channel order.
set(_gsplat_scheduled_channels ${_gsplat_final_num_channels})
list(SORT _gsplat_scheduled_channels COMPARE NATURAL ORDER DESCENDING)
foreach(_source IN LISTS _gsplat_instance_sources)
    foreach(_channel IN LISTS _gsplat_scheduled_channels)
        gsplat_target_add_source_instance(gsplat_core "${_source}" CHANNEL "${_channel}")
    endforeach()
endforeach()

# -----------------------------------------------------------------------------
# Python extension module
# -----------------------------------------------------------------------------

gsplat_add_python_extension(TARGET csrc SOURCES ext.cpp DESTINATION gsplat)
target_link_libraries(csrc PRIVATE gsplat_core)
set_target_properties(
    csrc
    PROPERTIES
        OUTPUT_NAME csrc
        LIBRARY_OUTPUT_DIRECTORY "${_gsplat_python_package_dir}"
        BUILD_RPATH_USE_ORIGIN ON
)

if(UNIX AND NOT APPLE)
    # Keep installed wheels relocatable. Torch and pip-provided CUDA runtime
    # libraries are siblings of the installed gsplat package; gsplat_core is a
    # private child of that package. Do not use INSTALL_RPATH_USE_LINK_PATH,
    # which would leak absolute build-environment paths into the wheel.
    set_target_properties(
        gsplat_core
        PROPERTIES INSTALL_RPATH "$ORIGIN/../../torch/lib;$ORIGIN/../../nvidia/cuda_runtime/lib"
    )
    set_target_properties(
        csrc
        PROPERTIES
            INSTALL_RPATH "$ORIGIN/lib;$ORIGIN/../torch/lib;$ORIGIN/../nvidia/cuda_runtime/lib"
    )
endif()

install(TARGETS gsplat_core LIBRARY DESTINATION gsplat/lib COMPONENT gsplat_runtime)
