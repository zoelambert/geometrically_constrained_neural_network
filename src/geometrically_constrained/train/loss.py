import torch
import torch.nn.functional as F
from torch import Tensor


def dice_loss(output: Tensor, target: Tensor) -> Tensor:
    """
    Calculate the Dice loss.

    Args:
        output (Tensor): Model predictions.
        target (Tensor): Ground truth labels.

    Returns:
        Tensor: Dice loss.
    """
    target = target.long()
    n, c, h, w = output.size()

    y_onehot = torch.zeros((n, c, h, w), device=output.device)
    y_onehot.scatter_(1, target.view(n, 1, h, w), 1)

    EPSILON = 1e-8
    probs = F.softmax(output, dim=1)[:, 1:]
    y_onehot = y_onehot[:, 1:]

    num = torch.sum(probs * y_onehot, dim=(2, 3))
    den = torch.sum(probs * probs + y_onehot * y_onehot, dim=(2, 3))
    dice = 4 - torch.sum((2 * num + EPSILON) / (den + EPSILON)) / n

    return dice


def geometric_loss(
    output: Tensor, target: Tensor, u: Tensor, w: Tensor, mu: float = 2
) -> Tensor:
    """
    Calculate the geometric loss.

    Args:
        output (Tensor): Model predictions.
        target (Tensor): Ground truth labels.
        u (Tensor): Auxiliary variable.
        w (Tensor): Langrangian variable.
        mu (float, optional): Weighting parameter. Defaults to 2.

    Returns:
        Tensor: Geometric loss.
    """
    target = target.long()
    n, c, h, l = output.size()

    y_onehot = torch.zeros((n, c, h, l), device=output.device)
    y_onehot.scatter_(1, target.view(n, 1, h, l), 1)

    EPSILON = 1e-8
    probs = F.softmax(output, dim=1)[:, 1:]
    y_onehot = y_onehot[:, 1:]

    num = torch.sum(probs * y_onehot, dim=(2, 3))
    den = torch.sum(probs * probs + y_onehot * y_onehot, dim=(2, 3))
    dice = 4 - torch.sum((2 * num + EPSILON) / (den + EPSILON)) / n

    geom = (mu / 2) * torch.sum((probs - u + w) * (probs - u + w)) / (n * h * l)

    region_in = torch.abs(torch.sum(probs * ((y_onehot - 1) ** 2)))
    region_out = torch.abs(torch.sum((1 - probs) * ((y_onehot) ** 2)))
    ms = (region_in + region_out) / (n * h * l)

    loss = geom + dice + ms

    return loss
