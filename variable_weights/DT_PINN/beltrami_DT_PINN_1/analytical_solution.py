from prm import a,d
import prm
import torch
from numpy_to_torch import pytorch_output

# u_func, v_func, w_func, p_func: analytical solutions of the Beltrami equations
# for any time instant

def u_func(x):
    x = pytorch_output(x)
    return (
        -a
        * (
            torch.exp(a * x[:, 0:1]) * torch.sin(a * x[:, 1:2] + d * x[:, 2:3])
            + torch.exp(a * x[:, 2:3]) * torch.cos(a * x[:, 0:1] + d * x[:, 1:2])
        )
        * torch.exp(-(d ** 2) * prm.time)
    )


def v_func(x):
    x = pytorch_output(x)
    return (
        -a
        * (
            torch.exp(a * x[:, 1:2]) * torch.sin(a * x[:, 2:3] + d * x[:, 0:1])
            + torch.exp(a * x[:, 0:1]) * torch.cos(a * x[:, 1:2] + d * x[:, 2:3])
        )
        * torch.exp(-(d ** 2) * prm.time)
    )


def w_func(x):
    x = pytorch_output(x)
    return (
        -a
        * (
            torch.exp(a * x[:, 2:3]) * torch.sin(a * x[:, 0:1] + d * x[:, 1:2])
            + torch.exp(a * x[:, 1:2]) * torch.cos(a * x[:, 2:3] + d * x[:, 0:1])
        )
        * torch.exp(-(d ** 2) * prm.time)
    )

def p_func(x):
    x = pytorch_output(x)
    return (
        -0.5
        * a ** 2
        * (
            torch.exp(2 * a * x[:, 0:1])
            + torch.exp(2 * a * x[:, 1:2])
            + torch.exp(2 * a * x[:, 2:3])
            + 2
            * torch.sin(a * x[:, 0:1] + d * x[:, 1:2])
            * torch.cos(a * x[:, 2:3] + d * x[:, 0:1])
            * torch.exp(a * (x[:, 1:2] + x[:, 2:3]))
            + 2
            * torch.sin(a * x[:, 1:2] + d * x[:, 2:3])
            * torch.cos(a * x[:, 0:1] + d * x[:, 1:2])
            * torch.exp(a * (x[:, 2:3] + x[:, 0:1]))
            + 2
            * torch.sin(a * x[:, 2:3] + d * x[:, 0:1])
            * torch.cos(a * x[:, 1:2] + d * x[:, 2:3])
            * torch.exp(a * (x[:, 0:1] + x[:, 1:2]))
        )
        * torch.exp(-2 * d ** 2 * prm.time)
    )

# u0_func, v0_func, w0_func, p0_func: analytical solutions of the Beltrami equations
# for t=0

def u0_func(x):
    x = pytorch_output(x)
    return (
        -a
        * (
            torch.exp(a * x[:, 0:1]) * torch.sin(a * x[:, 1:2] + d * x[:, 2:3])
            + torch.exp(a * x[:, 2:3]) * torch.cos(a * x[:, 0:1] + d * x[:, 1:2])
        )
    )


def v0_func(x):
    x = pytorch_output(x)
    return (
        -a
        * (
            torch.exp(a * x[:, 1:2]) * torch.sin(a * x[:, 2:3] + d * x[:, 0:1])
            + torch.exp(a * x[:, 0:1]) * torch.cos(a * x[:, 1:2] + d * x[:, 2:3])
        )
    )


def w0_func(x):
    x = pytorch_output(x)
    return (
        -a
        * (
            torch.exp(a * x[:, 2:3]) * torch.sin(a * x[:, 0:1] + d * x[:, 1:2])
            + torch.exp(a * x[:, 1:2]) * torch.cos(a * x[:, 2:3] + d * x[:, 0:1])
        )
    )

def p0_func(x):
    x = pytorch_output(x)
    return (
        -0.5
        * a ** 2
        * (
            torch.exp(2 * a * x[:, 0:1])
            + torch.exp(2 * a * x[:, 1:2])
            + torch.exp(2 * a * x[:, 2:3])
            + 2
            * torch.sin(a * x[:, 0:1] + d * x[:, 1:2])
            * torch.cos(a * x[:, 2:3] + d * x[:, 0:1])
            * torch.exp(a * (x[:, 1:2] + x[:, 2:3]))
            + 2
            * torch.sin(a * x[:, 1:2] + d * x[:, 2:3])
            * torch.cos(a * x[:, 0:1] + d * x[:, 1:2])
            * torch.exp(a * (x[:, 2:3] + x[:, 0:1]))
            + 2
            * torch.sin(a * x[:, 2:3] + d * x[:, 0:1])
            * torch.cos(a * x[:, 1:2] + d * x[:, 2:3])
            * torch.exp(a * (x[:, 0:1] + x[:, 1:2]))
        )
    )
