#!/bin/bash

# distributed setting, 'rank_node' should be different for each node when using more than one mechine (node)
master_addr=localhost
master_port=29500
gpu_per_node=8
num_nodes=1
rank_node=0

print_freq=5
local_lr=0.001
attack=blended              # badnet

epochs=100
batch_size=2000
physical_size=5000
lr=1
mgn=100000
augment_multiplicity=1
num_local_iter=5
eps=8
num_source=10
arch=preactresnet18         # preactresnet18 resnet20 WRN_16_4

echo "pernode4.sh is called"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096
python -u main.py  --task backdoor \
                   --epochs $epochs \
                   --batch-size $batch_size \
                   --physical-size $physical_size \
                   --lr $lr \
                   --mgn $mgn \
                   --augment-multiplicity $augment_multiplicity \
                   --num-local-iter $num_local_iter \
                   --gpu-per-node $gpu_per_node \
                   --num-nodes $num_nodes \
                   --rank-node $rank_node \
                   --master-addr $master_addr \
                   --master-port $master_port \
                    --eps $eps \
                    -p $print_freq \
                    --save-every 1 --workers 0 \
                    --arch $arch  \
                    --local-lr $local_lr --attack $attack --num-source $num_source --full-mode-do-clip \
                    --description backdoor_task \
                    --save-dir log_backdoor
