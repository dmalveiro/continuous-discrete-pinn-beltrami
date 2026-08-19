import numpy as np
import deepxde as dde
import ns_equations as nseq
import bc_ic
import param

def create_model(it):

    training_set = bc_ic.create_training_set(it)

    data = dde.data.TimePDE(
        bc_ic.spatio_temporal_domain,
        nseq.pde,
        [
           bc_ic.boundary_condition_u,
           bc_ic.boundary_condition_v,
           bc_ic.boundary_condition_w,
           bc_ic.initial_condition_u,
           bc_ic.initial_condition_v,
           bc_ic.initial_condition_w,
        ],
        num_domain=0,
        num_boundary=0,
        num_initial=0,
        anchors = training_set,
        num_test=None
    )

    net = dde.nn.FNN([param.inl] + param.n_hidl * [param.n_elem] + [param.outl], param.act_func, param.w_init)

    model = dde.Model(data, net)

#    import plots1
#    plots1.printall(data, model)

    return (data, model)
