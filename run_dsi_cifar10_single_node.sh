#!/bin/bash

# distributed setting, 'rank_node' should be different for each node when using more than one mechine (node)
master_addr=localhost
master_port=29500
gpu_per_node=8
num_nodes=1
rank_node=0

print_freq=1000
physical_size=8000
save_every=1

epochs=15
batch_size=20
lr=1
mgn=1
augment_multiplicity=2
num_local_iter=20
batch_enhancement=20
eps=8
arch=WRN_16_4
save_dir=log_cifar10

echo "run_cifar10.sh is called"
python -u main.py   --task cifar10 \
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
                    -r \
                    --save-every $save_every \
                    --batch-enhancement $batch_enhancement \
                    --arch $arch \
                    --save-dir $save_dir

