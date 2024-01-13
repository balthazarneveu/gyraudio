from gyraudio.audio_separation.metrics import Metrics
from gyraudio.audio_separation.properties import SIGNAL, NOISE
import torch


def test_metrics(device="cpu"):
    metric = Metrics("check")
    batch_size = 4
    gt_1 = torch.zeros(batch_size, 1, 512, device=device, requires_grad=True)
    gt_2 = torch.zeros(batch_size, 1, 256, device=device)
    for epoch in range(4):
        metric.reset_epoch()
        for step in range(10):
            # Prediction
            pred_1 = step*torch.randn(*gt_1.shape, device=device)
            pred_2 = epoch*torch.ones(*gt_2.shape, device=device)
            metric.update(pred_1, gt_1, SIGNAL)
            metric.update(pred_2, gt_2, NOISE)
            loss = metric.finish_step()
            # Backprop/update weights etc...
            loss.backward()
            print(f"epoch {epoch} | step {step} : {metric.metrics}")
        metric.finish_epoch()
        print(f"epoch {epoch} >>>>>>>> : {metric.total_metric}")
        print(metric)
