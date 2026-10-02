import unittest

import torch

from gsplat.losses import reduce_mean


class BroadcastMaskedMeanTest(unittest.TestCase):
    def test_shared_channel_mask_counts_every_included_element(self):
        values = torch.arange(24, dtype=torch.float64).reshape(2, 3, 2, 2)
        mask = torch.tensor([[[[True, False], [False, True]]],
                             [[[False, True], [True, False]]]])
        expected = values[mask.expand_as(values)].mean()
        torch.testing.assert_close(reduce_mean(values, mask), expected)

    def test_integer_weights_broadcast_in_numerator_and_denominator(self):
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        mask = torch.tensor([[2, 0, 1]])
        expected = (2 * 1 + 3 + 2 * 4 + 6) / (2 + 1 + 2 + 1)
        self.assertAlmostEqual(reduce_mean(values, mask).item(), expected)

    def test_excluded_nonfinite_values_do_not_change_mean_or_gradients(self):
        values = torch.tensor([[1.0, float("nan")], [3.0, float("inf")]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        loss = reduce_mean(values, mask)
        torch.testing.assert_close(loss, torch.tensor(2.0))
        gradient = torch.autograd.grad(loss, values)[0]
        torch.testing.assert_close(gradient, torch.tensor([[0.5, 0.0], [0.5, 0.0]]))

    def test_empty_broadcast_mask_returns_connected_zero(self):
        values = torch.full((2, 3), float("nan"), requires_grad=True)
        loss = reduce_mean(values, torch.zeros(1, 3, dtype=torch.bool))
        torch.testing.assert_close(loss, torch.tensor(0.0))
        gradient = torch.autograd.grad(loss, values)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(values))


if __name__ == "__main__":
    unittest.main()
