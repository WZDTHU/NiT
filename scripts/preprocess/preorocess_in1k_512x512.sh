NNODES = 1
GPUS_PER_NODE = 8
MASTER_ADDR = "localhost"
export MASTER_PORT=$((30000 + $RANDOM % 21000))

CMD=" \
    projects/preprocess/image_latent_c2i.py \
    --config configs/preprocess/imagenet1k_512x512.yaml \
    --project_dir workdir/preprocess/imagenet1k_512x512 \
    --seed 0 \
    "
TORCHLAUNCHER="torchrun \
    --nnodes $NNODES \
    --nproc_per_node $GPUS_PER_NODE \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    "
bash -c "$TORCHLAUNCHER $CMD"