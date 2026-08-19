import deepxde as dde
import numpy as np
from analytical_solution import u_func, v_func, w_func, p_func
import ns_equations as nseq
from param import n_test
import numpy_to_torch
import torch

def pred_exact(xt, model):

    # Convert xt to a PyTorch tensor and turn ON gradients for calculus
    xt = numpy_to_torch.pytorch_output(xt)
    xt.requires_grad_(True)

    # Predict <u>=(u,v,w,p) at the coordinates (x,y,z,t)
    output = model.net(xt)

    # <u>=(u,v,w,p) predictions at points xt=(x,y,z,t)
    u_pred = output[:, 0].reshape(-1)
    v_pred = output[:, 1].reshape(-1)
    w_pred = output[:, 2].reshape(-1)
    p_pred = output[:, 3].reshape(-1)

    # <u>=(u,v,w,p) exact results at points xt=(x,y,z,t)
    u_exact = u_func(xt).reshape(-1)
    v_exact = v_func(xt).reshape(-1)
    w_exact = w_func(xt).reshape(-1)
    p_exact = p_func(xt).reshape(-1)

    # Difference between the means of the predicted and exact pressures
    delta_p = torch.mean(p_pred) - torch.mean(p_exact)

    # Shift the prediction to align with the exact solution's mean
    p_pred_corrected = p_pred - delta_p

    # Predict the residuals at each mesh point stored at xt
    f = []
    nx, ny, nz = n_test, n_test, n_test

    for islc in range(ny):

        slice0 = islc * nx * nz
        slice1 = (islc + 1) * nx * nz
        xt_slice = xt[slice0:slice1, :]

        # Here, gradients are needed because pde function will be used
        xt_slice = xt_slice.clone().detach().requires_grad_(True)
        output_slice = model.net(xt_slice)

        # Predict the residuals at each mesh point stored at xt_slice
        f_slice = nseq.pde(xt_slice, output_slice)

        f_slice = torch.cat(f_slice, dim=1)
        f.append(f_slice.detach())
        dde.grad.clear()

    f = torch.cat(f, dim=0)

    # Detach, pass from GPU to CPU memory and convert from torch tensor to numpy array
    u_exact = u_exact.detach().cpu().numpy()
    u_pred = u_pred.detach().cpu().numpy()
    v_exact = v_exact.detach().cpu().numpy()
    v_pred = v_pred.detach().cpu().numpy()
    w_exact = w_exact.detach().cpu().numpy()
    w_pred = w_pred.detach().cpu().numpy()
    p_exact = p_exact.detach().cpu().numpy()
    p_pred_corrected = p_pred_corrected.detach().cpu().numpy()
    f = f.detach().cpu().numpy()

    # Compare the predicted outputs with the exact results (test) at these mesh points at time t
    # By computing the L2 relative error
    l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
    l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
    l2_difference_w = dde.metrics.l2_relative_error(w_exact, w_pred)
    l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred_corrected)
    residual = np.mean(np.absolute(f))      # mean of the absolute residuals at time t

    return (l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p, residual,
    u_pred, v_pred, w_pred, p_pred_corrected, u_exact, v_exact, w_exact, p_exact, f
    )
