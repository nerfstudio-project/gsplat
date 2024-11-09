import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import torch
from torch import nn


class iResNet(nn.Module):
    def __init__(self, input_num=2):
        super().__init__()
        self.batch_size = 16
        self.tol = 1e-6
        self.inp_size_linear = (input_num,)

        torch.manual_seed(0)
        nodes = [Ff.graph_inn.InputNode(*self.inp_size_linear, name="input")]
        for i in range(5):
            nodes.append(
                Ff.graph_inn.Node(
                    nodes[-1],
                    Fm.IResNetLayer,
                    {
                        "hutchinson_samples": 1,
                        "internal_size": 512,
                        "n_internal_layers": 4,
                    },
                    conditions=[],
                    name=f"i_resnet_{i}",
                )
            )
        nodes.append(Ff.graph_inn.OutputNode(nodes[-1], name="output"))
        self.i_resnet_linear = Ff.GraphINN(nodes, verbose=False)

        for node in self.i_resnet_linear.node_list:
            if isinstance(node.module, Fm.IResNetLayer):
                node.module.lipschitz_correction()

    def forward(self, rays, sensor_to_frustum=True):
        if sensor_to_frustum:
            return self.i_resnet_linear(rays, jac=False)[0]
        else:
            return self.i_resnet_linear(rays, rev=True, jac=False)[0]

    def test_inverse(self):
        x = torch.randn(self.batch_size, *self.inp_size_linear)
        x = x * torch.randn_like(x)
        x = x + torch.randn_like(x)

        y = self.i_resnet_linear(x, jac=False)[0]
        x_hat = self.i_resnet_linear(y, rev=True, jac=False)[0]
        print(x)
        print(x_hat)

        print("Check that inverse is close to input")
        assert torch.allclose(x, x_hat, atol=self.tol)
        print("Pass the test")

