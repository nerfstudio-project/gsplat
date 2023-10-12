import pytest
import torch


device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_cumulative_intersects():
    from gsplat.compute_cumulative_intersects import ComputeCumulativeIntersects

    torch.manual_seed(42)

    num_points = 100

    num_tiles_hit = torch.randint(
        0, 100, (num_points,), device=device, dtype=torch.int32
    )

    num_intersects, cum_tiles_hit = ComputeCumulativeIntersects.apply(
        num_points, num_tiles_hit
    )

    _cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    _num_intersects = _cum_tiles_hit[-1]

    torch.testing.assert_close(num_intersects, _num_intersects)
    torch.testing.assert_close(cum_tiles_hit, _cum_tiles_hit)


if __name__ == "__main__":
    test_cumulative_intersects()
