# Benchmarking Continuous and Discrete-Time PINNs on the Three-Dimensional Beltrami Flow

This repository contains the code and the data files containing the computational results reported at the document in annex.

## Overview

The work compares the continuous-time or vanilla PINNs (V-PINNs) and discrete-time PINNs (DT-PINNs), each of which with and without Adaptive Loss Weights (ALW) — fixed weights and variable weights, respectively — varying the initial learning rate and activation function, for solving the three-dimensional (3D) transient Navier-Stokes equations, using the Beltrami flow analytical solutions as references.

The experiments investigate model accuracy and convergence, 

## Directory Structure

    continuous-discrete-pinn-beltrami/
    ├── fixed_weights/
    │   ├── DT_PINN/
    │   │   ├── beltrami_DT_PINN_0/
    │   │   │   ├── RUN_1000_sigmoid_1e_4/
    │   │   │   │   └── CSV_post_processing/
    │   │   │   └── RUN_1000_tanh_1e_3/
    │   │   │       └── CSV_post_processing/
    │   │   ├── beltrami_DT_PINN_1/
    │   │   │   ├── RUN_1000_sigmoid_1e_5/
    │   │   │   │   └── CSV_post_processing/
    │   │   │   └── RUN_1000_tanh_1e_4/
    │   │   │       └── CSV_post_processing/
    │   │   ├── beltrami_DT_PINN_2/
    │   │   │   └── RUN_1000_tanh_1e_5/
    │   │   │       └── CSV_post_processing/
    │   │   └── beltrami_DT_PINN_3/
    │   │   │    └── RUN_1000_sigmoid_1e_3/
    │   │   │        └── CSV_post_processing/
    │   └── V_PINN/
    │       ├── beltrami_V_PINN_0/
    │       ├── beltrami_V_PINN_1/
    │       ├── beltrami_V_PINN_2/
    │       └── beltrami_V_PINN_3/
    ├── variable_weights/
    │   ├── DT_PINN/
    │   │   ├── beltrami_DT_PINN_0/
    │   │   ├── beltrami_DT_PINN_1/
    │   │   ├── beltrami_DT_PINN_2/
    │   │   └── beltrami_DT_PINN_3/
    │   └── V_PINN/
    │       ├── beltrami_V_PINN_0/
    │       ├── beltrami_V_PINN_1/
    │       ├── beltrami_V_PINN_2/
    │       └── beltrami_V_PINN_3/
    └── memory_usage/
    ├── README.md
    └── LICENSE

The directory structure reflects the procedure used for training the models, namely, the 24 model implementations and hyperparameter combinations (case studies) analysed, while maximizing the usage of the hardware resources available. 

- `fixed_weights/` contains experiments using fixed loss weights.
- `variable_weights/` contains experiments using adaptive/variable loss weights.
- `DT_PINN/` contains the discrete-time PINN implementations.
- `V_PINN/` contains the continuous-time (or vanilla) PINN implementations.
- `beltrami_*_0` through `beltrami_*_3` correspond to the four identical GPUs (GPU number 0 to 3) used to run four different case studies in parallel. Each of these folders contain the codes needed to run the experiments.
- `RUN_*` is a naming convention stating the hyperparameter combination used for each case study, from left to right in the folder naming: number of training points, activation function, initial learning rate. This is where the training sets and the training times for the experiments are stored.
- `CSV_post_processing/` contains the processed data directly used to generate the accuracy and convergence results presented in the report.
- `memory_usage/` contains the memory usage registrations, in steps of 10 ms, during training.
