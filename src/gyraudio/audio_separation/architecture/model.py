import torch


class SeparationModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def count_parameters(self) -> int:
        """Count the total number of parameters of the model

        Returns:
            int: total amount of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def receptive_field(self) -> int:
        """Compute the receptive field of the model

        Returns:
            int: receptive field
        """
        input_tensor = torch.rand(1, 1, 4096, requires_grad=True)
        out, out_noise = self.forward(input_tensor)
        grad = torch.zeros_like(out)
        grad[..., out.shape[-1]//2] = torch.nan  # set NaN gradient at the middle of the output
        out.backward(gradient=grad)
        self.zero_grad()  # reset to avoid future problems
        return int(torch.sum(input_tensor.grad.isnan()).cpu())  # Count NaN in the input
