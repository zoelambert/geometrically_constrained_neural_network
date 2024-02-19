import torch
from torch import Tensor

from geometrically_constrained.conf.conf_file import config


def div(Px: Tensor, Py: Tensor) -> Tensor:
    """
    Compute the divergence of a vector field.

    Args:
        Px (Tensor): X-component of the vector field.
        Py (Tensor): Y-component of the vector field.

    Returns:
        Tensor: Divergence of the vector field.
    """
    nx = Px.size(2)
    div_x = Px - torch.cat([Px[:, :, 0:1, :], Px[:, :, 0 : nx - 1, :]], dim=2)
    div_x[:, :, 0, :] = Px[:, :, 0, :]  # boundary conditions
    div_x[:, :, nx - 1, :] = -Px[:, :, nx - 2, :]

    ny = Py.size(3)
    div_y = Py - torch.cat([Py[:, :, :, 0:1], Py[:, :, :, 0 : ny - 1]], dim=3)
    div_y[:, :, :, 0] = Py[:, :, :, 0]  # boundary conditions
    div_y[:, :, :, ny - 1] = -Py[:, :, :, ny - 2]

    div = div_x + div_y

    return div


def grad(M: Tensor) -> Tensor:
    """
    Compute the gradient of a tensor.

    Args:
        M (Tensor): Input tensor.

    Returns:
        Tensor: Gradient of the input tensor.
    """
    grad_x = torch.zeros_like(M)
    grad_x[:, :, :-1] = M[:, :, 1:] - M[:, :, :-1]

    grad_y = torch.zeros_like(M)
    grad_y[:, :, :, :-1] = M[:, :, :, 1:] - M[:, :, :, :-1]

    grad = torch.cat((torch.unsqueeze(grad_x, 0), torch.unsqueeze(grad_y, 0)), dim=0)

    return grad


def prox_simplex(x: Tensor) -> Tensor:
    """
    Compute the proximal operator of the simplex constraint.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Proximal operator of the simplex constraint.
    """
    b_sorted, indices = torch.sort(x, dim=1, descending=True)
    cssv = torch.cumsum(b_sorted, dim=1)
    weights = (
        torch.arange(1, x.size(1) + 1, device=x.device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    b_weighted = b_sorted * weights
    rho = (((b_sorted > cssv - 1).sum(1)) - 1).unsqueeze(1)
    theta = (torch.gather(cssv, 1, rho) - 1) / (rho + 1).float()
    sol = torch.clamp((x - theta), min=0)

    return sol


def splitting_algo(
    outputs: Tensor,
    masks: Tensor,
    w: Tensor,
    mu: float = 0.5,
    tau: float = 0.4,
    sigma: float = 0.01,
) -> Tensor:
    """
    Apply the splitting algorithm.

    Args:
        outputs (Tensor): Model predictions.
        masks (Tensor): Ground truth labels.
        w (Tensor): Lagrangian variable.
        mu (float, optional): Weighting parameter. Defaults to 0.5.
        tau (float, optional): Parameter. Defaults to 0.4.
        sigma (float, optional): Parameter. Defaults to 0.01.

    Returns:
        Tensor: Result of the splitting algorithm.
    """
    n, c, h, l = outputs.shape

    # Edge function detector to compute weight g of Tvg
    y_onehot = torch.zeros((n, c, h, l), device=outputs.device)
    y_onehot.scatter_(1, masks.view(n, 1, h, l), 1)
    dx_mask = (
        y_onehot[:, :, torch.cat([torch.arange(1, h), torch.tensor([h - 1])], dim=0), :]
        - y_onehot
    )
    dy_mask = (
        y_onehot[:, :, :, torch.cat([torch.arange(1, l), torch.tensor([l - 1])], dim=0)]
        - y_onehot
    )
    weight_g = 1e2 * torch.abs(dx_mask**2 + dy_mask**2)
    weight_g = 1 / (1 + weight_g)

    # Compute mask volume
    alpha = torch.sum(y_onehot, dim=(2, 3))

    # Initialization
    w_n = torch.zeros((n, c, h, l), device=outputs.device)
    w_n[:, 1:] = w
    w_n[:, 0] = -torch.sum(w_n[:, 1:], dim=1)
    s = torch.softmax(outputs, 1)
    u = s
    v = s
    p = grad(u)
    u_bar = torch.zeros_like(s)

    for i in range(config["primal_dual"]["nb_iter"]):
        # Update p
        z = p + sigma * grad(u_bar)
        p = z / torch.clamp(((((z**2).sum(0)) ** 0.5) / weight_g), min=1)
        # Update v
        a = (
            ((alpha - ((1 - tau) * v + tau * u).sum(3).sum(2)) / (h * l))
            .unsqueeze(2)
            .unsqueeze(3)
        )
        v = ((1 - tau) * v + tau * u) + a
        # Update u, ubar
        uprevious = u
        x = (tau * mu * (s + w_n) + tau * (v + div(p[0], p[1]) - u) + u) * (
            1 / (tau * mu + 1)
        )
        u = prox_simplex(x)
        u_bar = 2 * u - uprevious

    w = w + (s - u)[:, 1:]
    return u[:, 1:], w
