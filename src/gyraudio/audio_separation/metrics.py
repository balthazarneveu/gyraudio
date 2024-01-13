from gyraudio.audio_separation.properties import SIGNAL, NOISE, TOTAL, LOSS_TYPE, COEFFICIENT
import torch

DEFAULT_COST = {
    SIGNAL: {
        COEFFICIENT: 0.5,
        LOSS_TYPE: torch.nn.functional.mse_loss
    },
    NOISE: {
        COEFFICIENT: 0.5,
        LOSS_TYPE: torch.nn.functional.mse_loss
    }
}


class Metrics:
    def __init__(self, name, costs=DEFAULT_COST) -> None:
        self.name = name
        self.keys = list(costs.keys())
        self.total_coefficient = sum([costs[key][COEFFICIENT] for key in self.keys])
        self.cost = costs
        self.reset_epoch()

    def reset_step(self):
        self.metrics = {key: 0. for key in self.keys}

    def reset_epoch(self):
        self.reset_step()
        self.total_metric = {key: 0. for key in self.keys+[TOTAL]}
        self.count = 0

    def update(self,
               prediction: torch.Tensor,
               ground_truth: torch.Tensor,
               key: str
               ):
        assert key != TOTAL
        loss_signal = self.cost[key][LOSS_TYPE](prediction, ground_truth)
        self.metrics[key] = loss_signal

    def finish_step(self):
        self.metrics[TOTAL] = 0.
        # Sum all metrics to total
        for key in self.metrics:
            if key != TOTAL:
                self.metrics[TOTAL] += self.cost[key][COEFFICIENT]*self.metrics[key]
        loss_signal = self.metrics[TOTAL]
        for key in self.metrics:
            self.metrics[key] = self.metrics[key].item()
            self.total_metric[key] += self.metrics[key]
        self.count += 1
        return loss_signal

    def finish_epoch(self):
        for key in self.metrics:
            self.total_metric[key] /= self.count
