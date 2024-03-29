from gyraudio.audio_separation.properties import SIGNAL, NOISE, TOTAL, LOSS_TYPE, COEFFICIENT, SNR
import torch


def snr(prediction: torch.Tensor, ground_truth: torch.Tensor, reduce="mean") -> torch.Tensor:
    """Compute the SNR between two tensors.
    Args:
        prediction (torch.Tensor): prediction tensor
        ground_truth (torch.Tensor): ground truth tensor
    Returns:
        torch.Tensor: SNR
    """
    power_signal = torch.sum(ground_truth**2, dim=(-2, -1))
    power_error = torch.sum((prediction-ground_truth)**2, dim=(-2, -1))
    eps = torch.finfo(torch.float32).eps
    snr_per_element = 10*torch.log10((power_signal+eps)/(power_error+eps))
    final_snr = torch.mean(snr_per_element) if reduce == "mean" else snr_per_element
    return final_snr


DEFAULT_COST = {
    SIGNAL: {
        COEFFICIENT: 0.5,
        LOSS_TYPE: torch.nn.functional.mse_loss
    },
    NOISE: {
        COEFFICIENT: 0.5,
        LOSS_TYPE: torch.nn.functional.mse_loss
    },
    SNR: {
        LOSS_TYPE: snr
    }
}


class Costs:
    """Keep track of cost functions.
    ```
    for epoch in range(...):
        metric.reset_epoch()
        for step in dataloader(...):
            ... # forward
            prediction = model.forward(batch)
            metric.update(prediction1, groundtruth1, SIGNAL1)
            metric.update(prediction2, groundtruth2, SIGNAL2)
            loss = metric.finish_step()

            loss.backward()
            ... # backprop
        metric.finish_epoch()
        ... # log metrics
    ```
    """

    def __init__(self, name: str, costs=DEFAULT_COST) -> None:
        self.name = name
        self.keys = list(costs.keys())
        self.cost = costs

    def __reset_step(self) -> None:
        self.metrics = {key: 0. for key in self.keys}

    def reset_epoch(self) -> None:
        self.__reset_step()
        self.total_metric = {key: 0. for key in self.keys+[TOTAL]}
        self.count = 0

    def update(self,
               prediction: torch.Tensor,
               ground_truth: torch.Tensor,
               key: str
               ) -> torch.Tensor:
        assert key != TOTAL
        # Compute loss for a single batch (=step)
        loss_signal = self.cost[key][LOSS_TYPE](prediction, ground_truth)
        self.metrics[key] = loss_signal

    def finish_step(self) -> torch.Tensor:
        # Reset current total
        self.metrics[TOTAL] = 0.
        # Sum all metrics to total
        for key in self.metrics:
            if key != TOTAL and self.cost[key].get(COEFFICIENT, False):
                self.metrics[TOTAL] += self.cost[key][COEFFICIENT]*self.metrics[key]
        loss_signal = self.metrics[TOTAL]
        for key in self.metrics:
            if not isinstance(self.metrics[key], float):
                self.metrics[key] = self.metrics[key].item()
            self.total_metric[key] += self.metrics[key]
        self.count += 1
        return loss_signal

    def finish_epoch(self) -> None:
        for key in self.metrics:
            self.total_metric[key] /= self.count

    def __repr__(self) -> str:
        rep = f"{self.name}\t:\t"
        for key in self.total_metric:
            rep += f"{key}: {self.total_metric[key]:.3e} | "
        return rep
