import torch


def fixed_point_solve(
    func,
    z0,
    max_iter=50,
    tolerance=1e-5
):
    """
    Stable fixed-point iteration for DEQ.

    Solves:
        z = f(z)

    using:
        z_{k+1} = f(z_k)
    """

    z = z0

    for iteration in range(1, max_iter + 1):

        z_next = func(z)

        residual = torch.max(
            torch.abs(z_next - z)
        ).item()

        z = z_next

        if residual < tolerance:
            return z, iteration, True

    return z, max_iter, False