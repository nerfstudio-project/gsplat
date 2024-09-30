import torch

from ..cuda._wrapper import selective_adam_update


class SelectiveAdam(torch.optim.Adam):
    def __init__(self, params, eps, betas):
        super().__init__(params=params, eps=eps, betas=betas)

    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            M = param.numel() // N

            selective_adam_update(
                param,
                param.grad,
                exp_avg,
                exp_avg_sq,
                visibility,
                lr,
                beta1,
                beta2,
                eps,
                N,
                M,
            )
