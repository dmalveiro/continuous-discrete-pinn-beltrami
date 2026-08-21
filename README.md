# Benchmarking Continuous and Discrete-Time PINNs on the Three-Dimensional Beltrami Flow

This repository contains the code and the data files containing the computational results reported at the document in annex, titled **Benchmarking Continuous and Discrete-Time PINNs on the Three-Dimensional Beltrami Flow**. This document also presents a detailed explanation on the algorithms and methodology followed. It is advised to read it for a full understanding of the project here published.

## Overview

The work compares the continuous-time or vanilla PINNs (V-PINNs) with discrete-time PINNs (DT-PINNs), each of which with and without Adaptive Loss Weights (ALW) — variable weights and fixed weights, respectively — for solving the three-dimensional (3D) transient Navier-Stokes equations, using the Beltrami flow analytical solutions as references.

The experiments investigate model accuracy, convergence and performance across the case studies considered.

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
    ├── memory_usage/
    ├── model_MOD.py
    ├── model_ORIGINAL.py
    ├── tables_report.xlsx
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

For example, the data files stored at `continuous-discrete-pinn-beltrami/V_PINN/beltrami_V_PINN_1/RUN_1000_sigmoid_1e_5` correspond to the case study where the model was trained with vanilla PINNs, using GPU number 1, with sigmoid activation function and initial learning rate equal to $10^{-5}$.

## Correspondence between results from the report and the data files

Figure 2:
- `continuous-discrete-pinn-beltrami/*_weights/V_PINN/beltrami_V_PINN_*/RUN_1000_*/CSV_post_processing/min_max_total_losses.csv`
- `continuous-discrete-pinn-beltrami/*_weights/V_PINN/beltrami_V_PINN_*/RUN_1000_*/CSV_post_processing/avg_loss_components.csv`
  
Figure 3:
- `continuous-discrete-pinn-beltrami/*_weights/V_PINN/beltrami_V_PINN_*/RUN_1000_*/CSV_post_processing/avg_res_l2errors_v2_time.csv`

Figure 4:
- `continuous-discrete-pinn-beltrami/*_weights/DT_PINN/beltrami_DT_PINN_*/RUN_1000_*/CSV_post_processing/avg_loss_components_ts0.csv`

**Note:** for the red/blue line, any file `avg_loss_components_ts0.csv` from any case study with DT-PINNs and tanh/sigmoid is suitable, since the files are all equal.

Figure 5:
- `continuous-discrete-pinn-beltrami/*_weights/DT_PINN/beltrami_DT_PINN_*/RUN_1000_*/CSV_post_processing/avg_loss_components.csv`

Figure 6:
- `continuous-discrete-pinn-beltrami/*_weights/DT_PINN/beltrami_DT_PINN_1/RUN_1000_sigmoid_1e_5/CSV_post_processing/avg_loss_components.csv`
- `continuous-discrete-pinn-beltrami/*_weights/DT_PINN/beltrami_DT_PINN_2/RUN_1000_tanh_1e_5/CSV_post_processing/avg_loss_components.csv`
  
Figure 7:
- `continuous-discrete-pinn-beltrami/*_weights/DT_PINN/beltrami_DT_PINN_*/RUN_1000_*/CSV_post_processing/avg_res_l2errors_v2_time.csv`
  
Figure 8:
- `continuous-discrete-pinn-beltrami/fixed_weights/V_PINN/beltrami_V_PINN_3/RUN_1000_sigmoid_1e_3/CSV_post_processing/avg_res_l2errors_v2_time.csv`
- `continuous-discrete-pinn-beltrami/variable_weights/DT_PINN/beltrami_DT_PINN_0/RUN_1000_sigmoid_1e_4/CSV_post_processing/avg_res_l2errors_v2_time.csv`
- `continuous-discrete-pinn-beltrami/variable_weights/DT_PINN/beltrami_DT_PINN_2/RUN_1000_tanh_1e_5/CSV_post_processing/avg_res_l2errors_v2_time.csv`
- `continuous-discrete-pinn-beltrami/variable_weights/DT_PINN/beltrami_DT_PINN_1/RUN_1000_sigmoid_1e_5/CSV_post_processing/avg_res_l2errors_v2_time.csv`

Table II:
- `continuous-discrete-pinn-beltrami/*_weights/V_PINN/beltrami_V_PINN_*/RUN_1000_*/CSV_post_processing/avg_res_l2errors_v2_time.csv`
- `continuous-discrete-pinn-beltrami/tables_report.xlsx` (tab "V_PINN")

Table III:
- `continuous-discrete-pinn-beltrami/*_weights/DT_PINN/beltrami_DT_PINN_*/RUN_1000_*/CSV_post_processing/avg_res_l2errors_v2_time.csv`
- `continuous-discrete-pinn-beltrami/tables_report.xlsx` (tab "DT_PINN")

Table IV:
- `continuous-discrete-pinn-beltrami/memory_usage/vram_*_vpinn_*.csv`
- `continuous-discrete-pinn-beltrami/tables_report.xlsx` (tab "performance")

Table V:
- `continuous-discrete-pinn-beltrami/memory_usage/vram_*_dtpinn_*.csv`
- `continuous-discrete-pinn-beltrami/tables_report.xlsx` (tab "performance")

## Hardware and Software setup

The code was developed and the experiments were performed using the configuration below:

- **Architecture:** x86-64 (virtualized)
- **CPU model:** Intel Xeon Silver 4214R
- **Memory:** 256 GB RAM
- **GPU:** Nvidia Tesla V100S PCIe 32 GB
- **Nvidia driver version:** 555.42.02
- **CUDA version:** 12.6
- **Operating System:** Ubuntu 20.04.6 LTS
- **ML Backend:** PyTorch 2.9.1+cu126
- **PINNs Framework:** DeepXDE 1.15.0

The computations for the present work are performed in single precision, using a single GPU for training and inferencing.

## The codes

This project uses the [DeepXDE](https://github.com/lululxvi/deepxde) framework. The model.py file from DeepXDE source code is modified (see `model_MOD.py`) to implement the adaptive loss weighting algorithm.

The implementation is based on the [Beltrami Flow example](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/Beltrami_flow.html) provided in the DeepXDE documentation, with substantial modifications to accomodate the project requirements. These can be seen in every `beltrami_*_0` through `beltrami_*_3` folder. The code logic is presented in the scheme below.

Thus, this project starts from an existing implementation, which was then transformed to support methodologies that are were not supported by that implementation.
