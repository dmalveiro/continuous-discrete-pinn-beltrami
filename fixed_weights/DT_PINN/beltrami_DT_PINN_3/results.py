import deepxde as dde
import numpy as np
from analytical_solution import u_func, v_func, w_func, p_func
import ns_equations as nseq
import prm
from prm import n_test, device, precision
import previous_model_grad
import numpy_to_torch
import torch

def create_mesh():

    # Create a n_test^3 mesh, with uniformly spaced mesh points
    # x=y=z=(n_test,n_test,n_test)
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, n_test), np.linspace(-1, 1, n_test), np.linspace(-1, 1, n_test)
    )

    # Matrix (1000,3): each line is the (x,y,z) coordinates of a mesh point
    xyz = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T

    # Optional: save test set (xyz) in a csv file
#    xyz_np = xyz.detach().cpu().numpy()
#    np.savetxt('xyz.csv', xyz, delimiter=',')

    return xyz


def pred_exact(xyz, previous_model, current_model, time_step):

    # Convert xyz to a PyTorch tensor
    xyz = numpy_to_torch.pytorch_output(xyz)

    # Efficient prediction, without gradients (do not need them because it is forward pass)
    # Predict <u>=(u,v,w,p) at the coordinates (x,y,z)
    current_model.net.eval()
    with torch.no_grad(): output_n1 = current_model.net(xyz)

    # <u>=(u,v,w,p) predictions at points xyz=(x,y,z)
    u_pred = output_n1[:, 0].reshape(-1)
    v_pred = output_n1[:, 1].reshape(-1)
    w_pred = output_n1[:, 2].reshape(-1)
    p_pred = output_n1[:, 3].reshape(-1)

    # <u>=(u,v,w,p) exact results at points xyz=(x,y,z)
    u_exact = u_func(xyz).reshape(-1)
    v_exact = v_func(xyz).reshape(-1)
    w_exact = w_func(xyz).reshape(-1)
    p_exact = p_func(xyz).reshape(-1)

    # Difference between the means of the predicted and exact pressures
    delta_p = torch.mean(p_pred) - torch.mean(p_exact)

    # Shift the prediction to align with the exact solution's mean
    p_pred_corrected = p_pred - delta_p

    if time_step > 0:

        f = []
        nx, ny, nz = n_test, n_test, n_test

        for islc in range(ny):

            slice0 = islc * nx * nz
            slice1 = (islc + 1) * nx * nz
            xyz_slice = xyz[slice0:slice1, :]

            # Here, gradients are needed because pde function will be used
            xyz_slice = xyz_slice.clone().detach().requires_grad_(True)
            output_n1_slice = current_model.net(xyz_slice)

            # Predict the residuals at each mesh point stored at xyz_slice
            prm.gg = 1
            output_n_slice = previous_model_grad.compute_gradients(xyz_slice, previous_model)
            f_slice = nseq.pde(xyz_slice, output_n1_slice, output_n_slice)
            prm.gg = 0

            f_slice = torch.cat(f_slice, dim=1)
            f.append(f_slice.detach())
            dde.grad.clear()

        f = torch.cat(f, dim=0)

    elif time_step == 0: f = 0

    # Detach, pass from GPU to CPU memory and convert from torch tensor to numpy array
    u_exact = u_exact.detach().cpu().numpy()
    u_pred = u_pred.detach().cpu().numpy()
    v_exact = v_exact.detach().cpu().numpy()
    v_pred = v_pred.detach().cpu().numpy()
    w_exact = w_exact.detach().cpu().numpy()
    w_pred = w_pred.detach().cpu().numpy()
    p_exact = p_exact.detach().cpu().numpy()
    p_pred_corrected = p_pred_corrected.detach().cpu().numpy()
    if time_step > 0: f = f.detach().cpu().numpy()

    # Compare the predicted outputs with the exact results (test) at these mesh points at time t
    # By computing the L2 relative error
    l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
    l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
    l2_difference_w = dde.metrics.l2_relative_error(w_exact, w_pred)
    l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred_corrected)
    if time_step > 0:
        residual = np.mean(np.absolute(f))      # mean of the absolute residuals at time t
    elif time_step == 0:
        residual = 0

    return (l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p, residual,
    u_pred, v_pred, w_pred, p_pred_corrected, u_exact, v_exact, w_exact, p_exact, f
    )
