NNODES = 1
GPUS_PER_NODE = 1
MASTER_ADDR = "localhost"
export MASTER_PORT=60563
mkdir -p workdir/c2i/nit_b_pack_merge_radio_65536
CMD=" \
    projects/train/packed_trainer_c2i.py \
    --config configs/c2i/nit_b_pack_merge_radio_65536.yaml \
    --project_dir workdir/c2i/nit_b_pack_merge_radio_65536 \
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