#include <ATen/core/Tensor.h>
#include <torch/types.h>
#include "Lidars.cuh"
#include "Ops.h"

RowOffsetStructuredSpinningLidarModelParametersExtDevice::RowOffsetStructuredSpinningLidarModelParametersExtDevice(
    const gsplat::RowOffsetStructuredSpinningLidarModelParametersExt &params)
    : n_rows{static_cast<int>(params.row_elevations_rad.size(0))}
    , n_columns{static_cast<int>(params.column_azimuths_rad.size(0))}
    , fov_vert_rad{params.fov_vert_rad}
    , fov_horiz_rad{params.fov_horiz_rad}
    , fov_eps_rad{params.fov_eps_rad}
    , spinning_direction{params.spinning_direction}
    , spinning_frequency_hz{params.spinning_frequency_hz}
    , angles_to_columns_map{params.angles_to_columns_map.data_ptr<int32_t>()}
    , map_dim{static_cast<int>(params.angles_to_columns_map.size(1)), // .x=width
              static_cast<int>(params.angles_to_columns_map.size(0))} // .y=height
{
    CHECK_INPUT(params.angles_to_columns_map);

    TORCH_CHECK(params.angles_to_columns_map.size(0) > 1 && params.angles_to_columns_map.size(1) > 1,
                "angles_to_columns_map dimensions must be > 1");

    this->map_resolution_rad = {
        params.fov_horiz_rad->span/(params.angles_to_columns_map.size(1)-1),
        params.fov_vert_rad->span/(params.angles_to_columns_map.size(0)-1)
    };

    if(params.angles_to_columns_map.dtype() != torch::kInt32)
    {
        throw std::invalid_argument("angles_to_columns_map dtype must be torch.int32");
    }
}
