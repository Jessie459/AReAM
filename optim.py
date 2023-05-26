import numpy as np
import torch


def cosine_schedule(base_lr, start_lr, final_lr, total_steps, warmup_steps=0):
    if warmup_steps > 0:
        warmup_schedule = np.linspace(start_lr, base_lr, warmup_steps)
    else:
        warmup_schedule = np.array([])

    steps = np.arange(total_steps - warmup_steps)
    schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * steps / len(steps)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_steps
    return schedule


class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)
        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group["lr"] for group in self.param_groups]

    def step(self, closure=None):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]["lr"] = self.__initial_lr[i] * lr_mult

        super().step(closure)
        self.global_step += 1
