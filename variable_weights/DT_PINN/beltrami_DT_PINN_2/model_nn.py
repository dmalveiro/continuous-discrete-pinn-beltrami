import numpy as np
import torch
import deepxde as dde
import ns_equations as nseq
import bc_ic
import prm
import previous_model_grad
import analytical_solution as asol
import archs

# For model A, the "PDE" is simply fitting the function at t=0
def pde_initial(x, y):

    res_u = y[:, 0:1] - asol.u0_func(x)
    res_v = y[:, 1:2] - asol.v0_func(x)
    res_w = y[:, 2:3] - asol.w0_func(x)
    res_p = y[:, 3:4] - asol.p0_func(x)

    return [res_u, res_v, res_w, res_p]

# Fourier embeddings (gamma = (n_train, embed_dim)), sigma = embed_scale**2
gamma = archs.FourierEmbs(input_dim=prm.inl, embed_dim=prm.inl_fourier, embed_scale=prm.sigma)      # leave uncommented if using fourier embeddings

def create_model_a(it):	# t = 0 -----------------------

    training_set = bc_ic.create_training_set(it)

    data_a = dde.data.PDE(
        bc_ic.spatial_domain,
        pde_initial,
        [],
        num_domain = 0,
        num_boundary = 0,
        anchors = training_set,
        num_test = None
    )

    net_a = dde.nn.FNN([prm.inl] + prm.n_hidl * [prm.n_elem] + [prm.outl], prm.act_func, prm.w_init)      		# standard network
##    net_a = dde.nn.FNN([prm.inl_fourier] + prm.n_hidl * [prm.n_elem] + [prm.outl], prm.act_func, prm.w_init)      	# only fourier embeddings
##    net_a = archs.ModifiedMLP([prm.inl] + prm.n_hidl * [prm.n_elem] + [prm.outl], prm.act_func, prm.w_init)      	# only modified MLP
##    net_a = archs.ModifiedMLP([prm.inl_fourier] + prm.n_hidl * [prm.n_elem] + [prm.outl], prm.act_func, prm.w_init)   # modified MLP + fourier embeddings

##    net_a.apply_feature_transform(gamma)      # leave uncommented if using fourier embeddings

    model_a = dde.Model(data_a, net_a)

    return (data_a, model_a)


def create_model_b(previous_model, it):	# t > 0 ----------------
    def aux_b(x):
        uN_n_mat = previous_model_grad.compute_gradients(x, previous_model)
        return uN_n_mat

    training_set = bc_ic.create_training_set(it)

    data_b = dde.data.PDE(
        bc_ic.spatial_domain,
        nseq.pde,						# Residual loss components
        [bc_ic.bc_u, bc_ic.bc_v, bc_ic.bc_w, bc_ic.bc_p],       # BC loss components
        num_domain = 0,
        num_boundary = 0,
        anchors = training_set,	                                # Training set
        num_test = None,
	auxiliary_var_function = aux_b			        # Results from the previous time step
    )

    net_b = dde.nn.FNN([prm.inl] + prm.n_hidl * [prm.n_elem] + [prm.outl], prm.act_func, prm.w_init)                    # standard network
##    net_b = dde.nn.FNN([prm.inl_fourier] + prm.n_hidl * [prm.n_elem] + [prm.outl], prm.act_func, prm.w_init)          # only fourier embeddings
##    net_b = archs.ModifiedMLP([prm.inl] + prm.n_hidl * [prm.n_elem] + [prm.outl], prm.act_func, prm.w_init)           # only modified MLP
##    net_b = archs.ModifiedMLP([prm.inl_fourier] + prm.n_hidl * [prm.n_elem] + [prm.outl], prm.act_func, prm.w_init)   # modified MLP + fourier embeddings

##    net_b.apply_feature_transform(gamma)      # leave uncommented if using fourier embeddings

    model_b = dde.Model(data_b, net_b)

    return (data_b, model_b)
