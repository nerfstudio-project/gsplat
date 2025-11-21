#import "bindings.h"
#import "config.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

struct MetalContext {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    dispatch_queue_t d_queue;

    id<MTLComputePipelineState> nd_rasterize_backward_kernel_cpso;
    id<MTLComputePipelineState> nd_rasterize_forward_kernel_cpso;
    id<MTLComputePipelineState> rasterize_backward_kernel_cpso;
    id<MTLComputePipelineState> project_gaussians_forward_kernel_cpso;
    id<MTLComputePipelineState> project_gaussians_backward_kernel_cpso;
    id<MTLComputePipelineState> compute_sh_forward_kernel_cpso;
    id<MTLComputePipelineState> compute_sh_backward_kernel_cpso;
    id<MTLComputePipelineState> compute_cov2d_bounds_kernel_cpso;
    id<MTLComputePipelineState> map_gaussian_to_intersects_kernel_cpso;
    id<MTLComputePipelineState> get_tile_bin_edges_kernel_cpso;
};

unsigned num_sh_bases(const unsigned degree) {
    if (degree == 0)
        return 1;
    if (degree == 1)
        return 4;
    if (degree == 2)
        return 9;
    if (degree == 3)
        return 16;
    return 25;
}

// This empty class lets us query for files relative to this file's bundle path using NSBundle bundleForClass hack
@interface DummyClassForPathHack : NSObject
@end
@implementation DummyClassForPathHack
@end

MetalContext* init_gsplat_metal_context() {
    MetalContext* ctx = (MetalContext*)malloc(sizeof(MetalContext));
    // Retrieve the default Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    // Configure context
    ctx->device = device;
    ctx->queue  = [ctx->device newCommandQueue];
    ctx->d_queue = torch::mps::get_dispatch_queue();

    NSError *error = nil;

    id<MTLLibrary> metal_library = nil;
    NSBundle * bundle = [NSBundle bundleForClass:[DummyClassForPathHack class]];
    NSString * path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];

    if (path_lib != nil) {
        // pre-compiled library found
        NSURL * libURL = [NSURL fileURLWithPath:path_lib];
        printf("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

        metal_library = [ctx->device newLibraryWithURL:libURL error:&error];
        if (error) {
            printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }
        printf("%s: loaded '%s', functions: %s\n", __func__, [path_lib UTF8String], [[[metal_library functionNames] componentsJoinedByString:@", "] UTF8String]);
    } else {
        printf("%s: default.metallib not found, loading from source\n", __func__);

        NSString * source_path = [[@ __FILE__ stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"gsplat_metal.metal"];
        printf("%s: loading '%s'\n", __func__, [source_path UTF8String]);

        NSString * src = [NSString stringWithContentsOfFile:source_path encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }

        @autoreleasepool {
            // dictionary of preprocessor macros
            NSMutableDictionary * prep = [NSMutableDictionary dictionary];

            MTLCompileOptions* options = [MTLCompileOptions new];
            options.preprocessorMacros = prep;

            metal_library = [ctx->device newLibraryWithSource:src options:options error:&error];
            if (error) {
                printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return NULL;
            }
        }
    }

#define GSPLAT_METAL_ADD_KERNEL(NAME) \
    { \
        id<MTLFunction> metal_function = [metal_library newFunctionWithName:@#NAME]; \
        printf("%s: load function %s with label: %s\n", __func__, #NAME, [[metal_function label] UTF8String]); \
        ctx->NAME ## _cpso = [ctx->device newComputePipelineStateWithFunction:metal_function error:&error]; \
        [metal_function release]; \
        if (error) { \
            printf("%s: error: load pipeline error: %s\n", __func__, [[error description] UTF8String]); \
            [metal_library release]; \
            return NULL; \
        } \
    }

    GSPLAT_METAL_ADD_KERNEL(nd_rasterize_backward_kernel);
    GSPLAT_METAL_ADD_KERNEL(nd_rasterize_forward_kernel);
    GSPLAT_METAL_ADD_KERNEL(rasterize_backward_kernel);
    GSPLAT_METAL_ADD_KERNEL(project_gaussians_forward_kernel);
    GSPLAT_METAL_ADD_KERNEL(project_gaussians_backward_kernel);
    GSPLAT_METAL_ADD_KERNEL(compute_sh_forward_kernel);
    GSPLAT_METAL_ADD_KERNEL(compute_sh_backward_kernel);
    GSPLAT_METAL_ADD_KERNEL(compute_cov2d_bounds_kernel);
    GSPLAT_METAL_ADD_KERNEL(map_gaussian_to_intersects_kernel);
    GSPLAT_METAL_ADD_KERNEL(get_tile_bin_edges_kernel);

    [metal_library release];

    return ctx;
}

MetalContext* get_global_context() {
    static MetalContext* ctx = NULL;
    if (ctx == NULL) {
        ctx = init_gsplat_metal_context();
    }
    return ctx;
}

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

#define ENC_SCALAR(encoder, x, i) [encoder setBytes:&x length:sizeof(x) atIndex:i]
#define ENC_ARRAY(encoder, x, i) [encoder setBytes:x length:sizeof(x) atIndex:i]
#define ENC_TENSOR(encoder, x, i) [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:i]

enum struct EncodeType {
    FLOAT,
    INT,
    UINT,
    ARRAY,
    TENSOR
};

struct EncodeArg {
    static EncodeArg scalar(float x) {
        return EncodeArg(EncodeType::FLOAT, x, 0, 0, nullptr, 0, nullptr);
    }
    static EncodeArg scalar(int32_t x) {
        return EncodeArg(EncodeType::INT, 0, x, 0, nullptr, 0, nullptr);
    }
    static EncodeArg scalar(uint32_t x) {
        return EncodeArg(EncodeType::UINT, 0, 0, x, nullptr, 0, nullptr);
    }
    static EncodeArg array(void* x, size_t numBytes) {
        return EncodeArg(EncodeType::ARRAY, 0, 0, 0, x, numBytes, nullptr);
    }
    static EncodeArg tensor(const torch::Tensor& x) {
        return EncodeArg(EncodeType::TENSOR, 0, 0, 0, nullptr, 0, &x);
    }
private:
    EncodeArg(
        EncodeType type,
        float fScalar,
        int32_t i32Scalar,
        uint32_t u32Scalar,
        void* array,
        size_t arrayNumBytes,
        const torch::Tensor* tensor
    ) : _type(type), _fScalar(fScalar), _i32Scalar(i32Scalar), _u32Scalar(u32Scalar), _array(array), _arrayNumBytes(arrayNumBytes), _tensor(tensor) {}
    EncodeType _type;
    float _fScalar;
    int32_t _i32Scalar;
    uint32_t _u32Scalar;
    void* _array;
    size_t _arrayNumBytes;
    const torch::Tensor* _tensor;

    friend void dispatchKernel(MetalContext* ctx, id<MTLComputePipelineState> cpso, MTLSize grid_size, MTLSize thread_group_size, std::vector<EncodeArg> args);
};

void dispatchKernel(MetalContext* ctx, id<MTLComputePipelineState> cpso, MTLSize grid_size, MTLSize thread_group_size, std::vector<EncodeArg> args) {
    // Get a reference to the command buffer for the MPS stream
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer, "Failed to retrieve command buffer reference");

    // Dispatch the kernel
    dispatch_sync(ctx->d_queue, ^(){
        // Start a compute pass
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        TORCH_CHECK(encoder, "Failed to create compute command encoder");

        // Encode the pipeline state object
        [encoder setComputePipelineState:cpso];

        // Encode arguments
        for (size_t i = 0; i < args.size(); ++i) {
            const EncodeArg& arg = args[i];
            switch (arg._type) {
                case EncodeType::FLOAT:
                    [encoder setBytes:&arg._fScalar length:sizeof(arg._fScalar) atIndex:i];
                    break;
                case EncodeType::INT:
                    [encoder setBytes:&arg._i32Scalar length:sizeof(arg._i32Scalar) atIndex:i];
                    break;
                case EncodeType::UINT:
                    [encoder setBytes:&arg._u32Scalar length:sizeof(arg._u32Scalar) atIndex:i];
                    break;
                case EncodeType::ARRAY:
                    [encoder setBytes:arg._array length:arg._arrayNumBytes atIndex:i];
                    break;
                case EncodeType::TENSOR:
                    [encoder setBuffer:getMTLBufferStorage(*arg._tensor) offset:arg._tensor->storage_offset() * arg._tensor->element_size() atIndex:i];
                    break;
            }
        }

        // Dispatch the compute command
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];

        // Commit the work
        torch::mps::synchronize();
    });
}

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii
compute_cov2d_bounds_tensor(const int num_pts, torch::Tensor &covs2d) {
    CHECK_INPUT(covs2d);
    torch::Tensor conics = torch::zeros(
        {num_pts, covs2d.size(1)}, covs2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor radii =
        torch::zeros({num_pts, 1}, covs2d.options().dtype(torch::kFloat32));

    // Dispatch the kernel
    MetalContext* ctx = get_global_context();
    MTLSize grid_size = MTLSizeMake(num_pts, 1, 1);
    NSUInteger num_threads_per_group =
        MIN(ctx->compute_cov2d_bounds_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_pts);
    MTLSize thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1);
    dispatchKernel(ctx, ctx->compute_cov2d_bounds_kernel_cpso, grid_size, thread_group_size, {
        EncodeArg::scalar(num_pts),
        EncodeArg::tensor(covs2d),
        EncodeArg::tensor(conics),
        EncodeArg::tensor(radii)
    });

    return std::make_tuple(conics, radii);
}

torch::Tensor compute_sh_forward_tensor(
    unsigned num_points,
    unsigned degree,
    unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &coeffs
) {
    unsigned num_bases = num_sh_bases(degree);
    if (coeffs.ndimension() != 3 || coeffs.size(0) != num_points ||
        coeffs.size(1) != num_bases || coeffs.size(2) != 3) {
        AT_ERROR("coeffs must have dimensions (N, D, 3)");
    }
    torch::Tensor colors = torch::empty({num_points, 3}, coeffs.options());

    // Dispatch the kernel
    MetalContext* ctx = get_global_context();
    MTLSize grid_size = MTLSizeMake(num_points, 1, 1);
    NSUInteger num_threads_per_group =
        MIN(ctx->compute_sh_forward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
    MTLSize thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1);
    dispatchKernel(ctx, ctx->compute_sh_forward_kernel_cpso, grid_size, thread_group_size, {
        EncodeArg::scalar(num_points),
        EncodeArg::scalar(degree),
        EncodeArg::scalar(degrees_to_use),
        EncodeArg::tensor(viewdirs),
        EncodeArg::tensor(coeffs),
        EncodeArg::tensor(colors)
    });
    return colors;
}

torch::Tensor compute_sh_backward_tensor(
    unsigned num_points,
    unsigned degree,
    unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
) {
    if (viewdirs.ndimension() != 2 || viewdirs.size(0) != num_points ||
        viewdirs.size(1) != 3) {
        AT_ERROR("viewdirs must have dimensions (N, 3)");
    }
    if (v_colors.ndimension() != 2 || v_colors.size(0) != num_points ||
        v_colors.size(1) != 3) {
        AT_ERROR("v_colors must have dimensions (N, 3)");
    }
    unsigned num_bases = num_sh_bases(degree);
    torch::Tensor v_coeffs =
        torch::zeros({num_points, num_bases, 3}, v_colors.options());

    // Dispatch the kernel
    MetalContext* ctx = get_global_context();
    MTLSize grid_size = MTLSizeMake(num_points, 1, 1);
    NSUInteger num_threads_per_group =
        MIN(ctx->compute_sh_backward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
    MTLSize thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1);
    dispatchKernel(ctx, ctx->compute_sh_backward_kernel_cpso, grid_size, thread_group_size, {
        EncodeArg::scalar(num_points),
        EncodeArg::scalar(degree),
        EncodeArg::scalar(degrees_to_use),
        EncodeArg::tensor(viewdirs),
        EncodeArg::tensor(v_colors),
        EncodeArg::tensor(v_coeffs)
    });

    return v_coeffs;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    torch::Tensor &viewmat,
    torch::Tensor &projmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    const std::tuple<int, int, int> tile_bounds,
    const float clip_thresh
) {
    // Triangular covariance.
    torch::Tensor cov3d_d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));

    float intrins[4] = {fx, fy, cx, cy};
    uint32_t img_size[2] = {img_width, img_height};
    uint32_t tile_bounds_arr[4] = {
        (uint32_t)std::get<0>(tile_bounds),
        (uint32_t)std::get<1>(tile_bounds),
        (uint32_t)std::get<2>(tile_bounds),
        0xDEAD
    };

    // Dispatch the kernel
    MetalContext* ctx = get_global_context();
    MTLSize grid_size = MTLSizeMake(num_points, 1, 1);
    NSUInteger num_threads_per_group =
        MIN(ctx->project_gaussians_forward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
    MTLSize thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1);
    dispatchKernel(ctx, ctx->project_gaussians_forward_kernel_cpso, grid_size, thread_group_size, {
        EncodeArg::scalar(num_points),
        EncodeArg::tensor(means3d),
        EncodeArg::tensor(scales),
        EncodeArg::scalar(glob_scale),
        EncodeArg::tensor(quats),
        EncodeArg::tensor(viewmat),
        EncodeArg::tensor(projmat),
        EncodeArg::array(intrins, sizeof(intrins)),
        EncodeArg::array(img_size, sizeof(img_size)),
        EncodeArg::array(tile_bounds_arr, sizeof(tile_bounds_arr)),
        EncodeArg::scalar(clip_thresh),
        EncodeArg::tensor(cov3d_d),
        EncodeArg::tensor(xys_d),
        EncodeArg::tensor(depths_d),
        EncodeArg::tensor(radii_d),
        EncodeArg::tensor(conics_d),
        EncodeArg::tensor(num_tiles_hit_d)
    });

    return std::make_tuple(
        cov3d_d, xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    torch::Tensor &viewmat,
    torch::Tensor &projmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    torch::Tensor &cov3d,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_depth,
    torch::Tensor &v_conic
) {
    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_cov3d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean3d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_scale =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_quat =
        torch::zeros({num_points, 4}, means3d.options().dtype(torch::kFloat32));

    float intrins[4] = {fx, fy, cx, cy};
    uint32_t img_size[2] = {img_width, img_height};

    MetalContext* ctx = get_global_context();
    MTLSize grid_size = MTLSizeMake(num_points, 1, 1);
    NSUInteger num_threads_per_group =
        MIN(ctx->project_gaussians_backward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
    MTLSize thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1);
    dispatchKernel(ctx, ctx->project_gaussians_backward_kernel_cpso, grid_size, thread_group_size, {
        EncodeArg::scalar(num_points),
        EncodeArg::tensor(means3d),
        EncodeArg::tensor(scales),
        EncodeArg::scalar(glob_scale),
        EncodeArg::tensor(quats),
        EncodeArg::tensor(viewmat),
        EncodeArg::tensor(projmat),
        EncodeArg::array(intrins, sizeof(intrins)),
        EncodeArg::array(img_size, sizeof(img_size)),
        EncodeArg::tensor(cov3d),
        EncodeArg::tensor(radii),
        EncodeArg::tensor(conics),
        EncodeArg::tensor(v_xy),
        EncodeArg::tensor(v_depth),
        EncodeArg::tensor(v_conic),
        EncodeArg::tensor(v_cov2d),
        EncodeArg::tensor(v_cov3d),
        EncodeArg::tensor(v_mean3d),
        EncodeArg::tensor(v_scale),
        EncodeArg::tensor(v_quat),
    });

    return std::make_tuple(v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat);
}


std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &num_tiles_hit,
    const std::tuple<int, int, int> tile_bounds
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(num_tiles_hit);

    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));

    uint32_t tile_bounds_arr[4] = {
        (uint32_t)std::get<0>(tile_bounds),
        (uint32_t)std::get<1>(tile_bounds),
        (uint32_t)std::get<2>(tile_bounds),
        0xDEAD
    };

    MetalContext* ctx = get_global_context();
    MTLSize grid_size = MTLSizeMake(num_points, 1, 1);
    NSUInteger num_threads_per_group =
        MIN(ctx->map_gaussian_to_intersects_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
    MTLSize thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1);
    dispatchKernel(ctx, ctx->map_gaussian_to_intersects_kernel_cpso, grid_size, thread_group_size, {
        EncodeArg::scalar(num_points),
        EncodeArg::tensor(xys),
        EncodeArg::tensor(depths),
        EncodeArg::tensor(radii),
        EncodeArg::tensor(num_tiles_hit),
        EncodeArg::array(tile_bounds_arr, sizeof(tile_bounds_arr)),
        EncodeArg::tensor(isect_ids_unsorted),
        EncodeArg::tensor(gaussian_ids_unsorted)
    });

    return std::make_tuple(isect_ids_unsorted, gaussian_ids_unsorted);
}

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects,
    const torch::Tensor &isect_ids_sorted
) {
    CHECK_INPUT(isect_ids_sorted);
    torch::Tensor tile_bins = torch::zeros(
        {num_intersects, 2}, isect_ids_sorted.options().dtype(torch::kInt32)
    );

    MetalContext* ctx = get_global_context();
    MTLSize grid_size = MTLSizeMake(num_intersects, 1, 1);
    NSUInteger num_threads_per_group =
        MIN(ctx->get_tile_bin_edges_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_intersects);
    MTLSize thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1);
    dispatchKernel(ctx, ctx->get_tile_bin_edges_kernel_cpso, grid_size, thread_group_size, {
        EncodeArg::scalar(num_intersects),
        EncodeArg::tensor(isect_ids_sorted),
        EncodeArg::tensor(tile_bins)
    });

    return tile_bins;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    // TODO(achan): we should be able to remove the 3rd dimension of `block` as it is always set to 1
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    const uint32_t channels = colors.size(1);
    const int img_width = std::get<0>(img_size);
    const int img_height = std::get<1>(img_size);

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    uint32_t img_size_dim3[4] = {(uint32_t)std::get<0>(img_size), (uint32_t)std::get<1>(img_size), (uint32_t)std::get<2>(img_size), 0xDEAD};
    uint32_t tile_bounds_arr[4] = {
        (uint32_t)std::get<0>(tile_bounds),
        (uint32_t)std::get<1>(tile_bounds),
        (uint32_t)std::get<2>(tile_bounds),
        0xDEAD
    };
    int32_t block_size_dim2[2] = {std::get<0>(block), std::get<1>(block)};

    MetalContext* ctx = get_global_context();
    MTLSize grid_size = MTLSizeMake(img_width, img_height, 1);
    MTLSize thread_group_size = MTLSizeMake(block_size_dim2[0], block_size_dim2[1], 1);
    dispatchKernel(ctx, ctx->nd_rasterize_forward_kernel_cpso, grid_size, thread_group_size, {
        EncodeArg::array(tile_bounds_arr, sizeof(tile_bounds_arr)),
        EncodeArg::array(img_size_dim3, sizeof(img_size_dim3)),
        EncodeArg::scalar(channels),
        EncodeArg::tensor(gaussian_ids_sorted),
        EncodeArg::tensor(tile_bins),
        EncodeArg::tensor(xys),
        EncodeArg::tensor(conics),
        EncodeArg::tensor(colors),
        EncodeArg::tensor(opacities),
        EncodeArg::tensor(final_Ts),
        EncodeArg::tensor(final_idx),
        EncodeArg::tensor(out_img),
        EncodeArg::tensor(background),
        EncodeArg::array(block_size_dim2, sizeof(block_size_dim2))
    });

    return std::make_tuple(out_img, final_Ts, final_idx);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> nd_rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    // TODO(achan): we should be able to remove the 3rd dimension of `block` as it is always set to 1
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    const uint32_t channels = colors.size(1);
    const int img_width = std::get<0>(img_size);
    const int img_height = std::get<1>(img_size);

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    uint32_t img_size_dim3[4] = {(uint32_t)std::get<0>(img_size), (uint32_t)std::get<1>(img_size), (uint32_t)std::get<2>(img_size), 0xDEAD};
    uint32_t tile_bounds_arr[4] = {
        (uint32_t)std::get<0>(tile_bounds),
        (uint32_t)std::get<1>(tile_bounds),
        (uint32_t)std::get<2>(tile_bounds),
        0xDEAD
    };
    int32_t block_size_dim2[2] = {std::get<0>(block), std::get<1>(block)};

    MetalContext* ctx = get_global_context();
    MTLSize grid_size = MTLSizeMake(img_width, img_height, 1);
    MTLSize thread_group_size = MTLSizeMake(block_size_dim2[0], block_size_dim2[1], 1);
    dispatchKernel(ctx, ctx->nd_rasterize_forward_kernel_cpso, grid_size, thread_group_size, {
        EncodeArg::array(tile_bounds_arr, sizeof(tile_bounds_arr)),
        EncodeArg::array(img_size_dim3, sizeof(img_size_dim3)),
        EncodeArg::scalar(channels),
        EncodeArg::tensor(gaussian_ids_sorted),
        EncodeArg::tensor(tile_bins),
        EncodeArg::tensor(xys),
        EncodeArg::tensor(conics),
        EncodeArg::tensor(colors),
        EncodeArg::tensor(opacities),
        EncodeArg::tensor(final_Ts),
        EncodeArg::tensor(final_idx),
        EncodeArg::tensor(out_img),
        EncodeArg::tensor(background),
        EncodeArg::array(block_size_dim2, sizeof(block_size_dim2))
    });

    return std::make_tuple(out_img, final_Ts, final_idx);
}


std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    nd_rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha
    ) {
    CHECK_INPUT(gaussians_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);
    CHECK_INPUT(final_Ts);
    CHECK_INPUT(final_idx);
    CHECK_INPUT(v_output);
    CHECK_INPUT(v_output_alpha);

    const int num_points = xys.size(0);
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());
    torch::Tensor workspace = torch::zeros(
        {img_height, img_width, channels},
        xys.options().dtype(torch::kFloat32)
    );

    // Get a reference to the command buffer for the MPS stream
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer, "Failed to retrieve command buffer reference");

    uint32_t img_size[2] = {img_width, img_height};
    uint32_t tile_bounds_arr[4] = {
        (img_width + BLOCK_X - 1) / BLOCK_X,
        (img_height + BLOCK_Y - 1) / BLOCK_Y,
        1,
        0xDEAD
    };

    MetalContext* ctx = get_global_context();
    MTLSize grid_size = MTLSizeMake(img_width, img_height, 1);
    MTLSize thread_group_size = MTLSizeMake(BLOCK_X, BLOCK_Y, 1);
    dispatchKernel(ctx, ctx->nd_rasterize_backward_kernel_cpso, grid_size, thread_group_size, {
        EncodeArg::array(tile_bounds_arr, sizeof(tile_bounds_arr)),
        EncodeArg::array(img_size, sizeof(img_size)),
        EncodeArg::scalar(channels),
        EncodeArg::tensor(gaussians_ids_sorted),
        EncodeArg::tensor(tile_bins),
        EncodeArg::tensor(xys),
        EncodeArg::tensor(conics),
        EncodeArg::tensor(colors),
        EncodeArg::tensor(opacities),
        EncodeArg::tensor(background),
        EncodeArg::tensor(final_Ts),
        EncodeArg::tensor(final_idx),
        EncodeArg::tensor(v_output),
        EncodeArg::tensor(v_output_alpha),
        EncodeArg::tensor(v_xy),
        EncodeArg::tensor(v_conic),
        EncodeArg::tensor(v_colors),
        EncodeArg::tensor(v_opacity),
        EncodeArg::tensor(workspace)
    });

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha
    ) {
    CHECK_INPUT(gaussians_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);
    CHECK_INPUT(final_Ts);
    CHECK_INPUT(final_idx);
    CHECK_INPUT(v_output);
    CHECK_INPUT(v_output_alpha);

    const int num_points = xys.size(0);
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());

    // Get a reference to the command buffer for the MPS stream
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer, "Failed to retrieve command buffer reference");

    uint32_t img_size[2] = {img_width, img_height};
    uint32_t tile_bounds_arr[4] = {
        (img_width + BLOCK_X - 1) / BLOCK_X,
        (img_height + BLOCK_Y - 1) / BLOCK_Y,
        1,
        0xDEAD
    };

    MetalContext* ctx = get_global_context();
    MTLSize grid_size = MTLSizeMake(img_width, img_height, 1);
    MTLSize thread_group_size = MTLSizeMake(BLOCK_X, BLOCK_Y, 1);
    dispatchKernel(ctx, ctx->rasterize_backward_kernel_cpso, grid_size, thread_group_size, {
        EncodeArg::array(tile_bounds_arr, sizeof(tile_bounds_arr)),
        EncodeArg::array(img_size, sizeof(img_size)),
        EncodeArg::tensor(gaussians_ids_sorted),
        EncodeArg::tensor(tile_bins),
        EncodeArg::tensor(xys),
        EncodeArg::tensor(conics),
        EncodeArg::tensor(colors),
        EncodeArg::tensor(opacities),
        EncodeArg::tensor(background),
        EncodeArg::tensor(final_Ts),
        EncodeArg::tensor(final_idx),
        EncodeArg::tensor(v_output),
        EncodeArg::tensor(v_output_alpha),
        EncodeArg::tensor(v_xy),
        EncodeArg::tensor(v_conic),
        EncodeArg::tensor(v_colors),
        EncodeArg::tensor(v_opacity)
    });

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}
