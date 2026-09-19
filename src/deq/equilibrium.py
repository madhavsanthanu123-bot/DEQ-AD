import torch
import torch.nn as nn
import torch.nn.functional as F

from .solver import fixed_point_solve


class DEQFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        input_weight,
        input_bias,
        hidden_weight,
        hidden_bias,
        beta,
        max_iter,
        tolerance
    ):
        """
        Forward pass of the Deep Equilibrium layer.

        Solves:

            z* = beta * tanh(W_h z* + W_x x + b)

        without storing the complete iteration history.
        """

        with torch.no_grad():

            x_features = F.linear(
                x,
                input_weight,
                input_bias
            )

            z = torch.zeros_like(x_features)

            def equilibrium_function(z_current):

                return beta * torch.tanh(
                    F.linear(
                        z_current,
                        hidden_weight,
                        hidden_bias
                    ) + x_features
                )

            z, iterations, converged = fixed_point_solve(
                equilibrium_function,
                z,
                max_iter=max_iter,
                tolerance=tolerance
            )

        ctx.save_for_backward(
            x,
            z,
            input_weight,
            input_bias,
            hidden_weight,
            hidden_bias
        )

        ctx.beta = beta
        ctx.max_iter = max_iter
        ctx.tolerance = tolerance

        DEQFunction.last_forward_iterations = iterations
        DEQFunction.last_forward_converged = converged

        return z

    @staticmethod
    def backward(ctx, grad_output):

        (
            x,
            z,
            input_weight,
            input_bias,
            hidden_weight,
            hidden_bias
        ) = ctx.saved_tensors

        beta = ctx.beta
        max_iter = ctx.max_iter
        tolerance = ctx.tolerance

        # IMPORTANT:
        # Reconstruct the COMPLETE graph inside enable_grad().
        with torch.enable_grad():

            x_recreated = x.detach().requires_grad_(True)
            z_recreated = z.detach().requires_grad_(True)

            x_features = F.linear(
                x_recreated,
                input_weight,
                input_bias
            )

            f_z = beta * torch.tanh(
                F.linear(
                    z_recreated,
                    hidden_weight,
                    hidden_bias
                ) + x_features
            )

            # -------------------------------------------------
            # Solve implicit adjoint equation:
            #
            # v = grad_output + J_f^T v
            # -------------------------------------------------

            v = grad_output.detach()

            backward_iterations = 0
            backward_converged = False

            for iteration in range(
                1,
                max_iter + 1
            ):

                jtv = torch.autograd.grad(
                    outputs=f_z,
                    inputs=z_recreated,
                    grad_outputs=v,
                    retain_graph=True,
                    create_graph=False
                )[0]

                v_next = grad_output + jtv

                residual = torch.max(
                    torch.abs(v_next - v)
                ).item()

                v = v_next

                backward_iterations = iteration

                if residual < tolerance:
                    backward_converged = True
                    break

            # -------------------------------------------------
            # Compute gradients with respect to:
            #
            # x
            # input weights
            # input bias
            # hidden weights
            # hidden bias
            # -------------------------------------------------

            gradients = torch.autograd.grad(
                outputs=f_z,
                inputs=(
                    x_recreated,
                    input_weight,
                    input_bias,
                    hidden_weight,
                    hidden_bias
                ),
                grad_outputs=v,
                retain_graph=False,
                create_graph=False,
                allow_unused=False
            )

        (
            grad_x,
            grad_input_weight,
            grad_input_bias,
            grad_hidden_weight,
            grad_hidden_bias
        ) = gradients

        DEQFunction.last_backward_iterations = (
            backward_iterations
        )

        DEQFunction.last_backward_converged = (
            backward_converged
        )

        return (
            grad_x,
            grad_input_weight,
            grad_input_bias,
            grad_hidden_weight,
            grad_hidden_bias,
            None,
            None,
            None
        )


class DEQLayer(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        beta=0.5,
        max_iter=50,
        tolerance=1e-5
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim
        )

        # Spectral normalization controls the recurrent
        # hidden transformation.
        self.hidden_layer = (
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(
                    hidden_dim,
                    hidden_dim
                )
            )
        )

        self.beta = beta
        self.max_iter = max_iter
        self.tolerance = tolerance

        self.last_iterations = 0
        self.last_converged = False

    def forward(self, x):

        z = DEQFunction.apply(
            x,
            self.input_projection.weight,
            self.input_projection.bias,
            self.hidden_layer.weight,
            self.hidden_layer.bias,
            self.beta,
            self.max_iter,
            self.tolerance
        )

        self.last_iterations = (
            DEQFunction.last_forward_iterations
        )

        self.last_converged = (
            DEQFunction.last_forward_converged
        )

        return z