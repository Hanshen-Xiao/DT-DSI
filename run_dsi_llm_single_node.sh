#!/bin/bash

# distributed setting, 'rank_node' should be different for each node when using more than one mechine (node)
master_addr=localhost
master_port=29500
gpu_per_node=8
num_nodes=1
rank_node=0

print_freq=500
epochs=5
batch_size=20
physical_size=1
lr=0.0000025
mgn=100000
augment_multiplicity=1
num_local_iter=1
eps=6
model_name=opt-350m #gpt2-medium #opt-350m #gpt2 #opt-125m
wiki_size=103
canary_repeat=1


echo "pernode_gpt.sh is called"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
python -u main.py --task llm \
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
                    --save-every 1 \
                    --add-canary --canary-repeat $canary_repeat --wiki-size $wiki_size --model-name-or-path $model_name  --canary-prompt wizard \
                    --wd 0 --block-size 1024 \
                    --description "${model_name}-wiki${wiki_size}-canrep${canary_repeat}-eps${eps}" \
                    --save-dir log_llm
