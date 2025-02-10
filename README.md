# Trustworthy Machine Learning through Data-Specific Indistinguishability

## Setting Up Environment for DSI

We include the codes in the paper "Trustworthy Machine Learning through Data-Specific Indistinguishability". 

You can run the following script to configure the necessary environment. Note that `requirements_1.txt` is sufficient for the `cifar10` and `llm` tasks, while `requirements_2.txt` is necessary for the `backdoor` task.

```
conda create --name dsi_env python=3.9
conda activate dsi_env
pip install -r requirements_1.txt
```

Due to the high computational requirements, we have only implemented the training algorithm using CUDA. Check `--save-dir/output_r0.txt` as the log file of the main process, and refer to other output files for logs of helper processes.
## Reproducing the `cifar10` Task Results

You can run the following script to train a model using the DSI approach with one node (mechine) equipped with 8 gpus:
```
    ./run_dsi_cifar10_single_node.sh
```
Please note that this task would take hours if the number of gpus is limited. 
To customize the training process or modify the arguments, please refer to [Clarification of Required Arguments](#clarification-of-required-arguments) and `parser_config.py` to edit `run_dsi_cifar10_single_node.sh`.

## Reproducing the `llm` Task Results

You can run the following script to train a model using the DSI approach with one node (mechine) equipped with 8 gpus:
```
    ./run_dsi_llm_single_node.sh
```
To customize the training process or modify the arguments, please refer to [Clarification of Required Arguments](#clarification-of-required-arguments) and `parser_config.py` to edit `run_dsi_cifar10_single_node.sh`.

## Reproducing the `backdoor` Task Results

You can run the following script to train a model using the DSI approach with one node (mechine) equipped with 8 gpus:
```
    ./run_dsi_backdoor_single_node.sh
```
To customize the training process or modify the arguments, please refer to [Clarification of Required Arguments](#clarification-of-required-arguments) and `parser_config.py` to edit `run_dsi_cifar10_single_node.sh`.

## Clarification of Required Arguments
- __`--task`__: Select a expiriment presented in paper. Choices = [`cifar10`, `llm`, `backdoor`]
### Multiple GPUs/Nodes Arguments
- __`--master-addr`__: IP address or hostname of the master node, which coordinates communication among worker nodes. Default value is `localhost`, which works when using one computer. __When running on multiple nodes, change it to the address of master node and pass a unique `--rank-node` to the script on each node in the range of the number of nodes, with the master node having `--rank-node=0`.__
- __`--master-port`__: Specifies a port number on the master node used for TCP-based communication between processes across different gpus or nodes. Default value is `29500`.
- __`--rank-node`__: Rank of current node (mechine).
- __`--gpu-per-node`__: Number of usable gpus on one node. Will use `cuda:0-gpu_per_node-1`.
### Training Arguments
- __`--epochs`__: Number of training epochs.
- __`--batch-size`__: Regular batch size for non `--full-mode`, subgroup size for `--full-mode`. Subgroup technique does not apply for `backdoor` task (only one subgroup).
- __`--physical-size`__: For `cifar10` task, set smaller when cuda memory is limited. for `llm` task, set to 1.
- __`--lr`__: Learning rate. Set to 1 for `--full-mode`.
- __`--mgn`__: Maximum gradient norm of (global) update clip. Set to a large number to disable clip.
- __`--num-local-iter`__: Automatically enable `--full-mode` when `--num-local-iter` is more than 1.
- __`--full-mode`__: Enable local updates. Will be set to `True` when `--num-local-iter` is more than 1.
- __`--batch-enhencement`__: Number of subgroups, only used in `cifar10` task.
- __`--eps`__: Trustworthy guarantee, choises = range(9) where 0 represent $+\infty$ (no noise) for convenience.
- __`--r`__: Enable resuming checkpoint from `--save-dir`.
- __`--save-dir`__: Where log and checkpoint are placed.

