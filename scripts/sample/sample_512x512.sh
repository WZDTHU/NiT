torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  projects/sample/sample_c2i_ddp.py \
  --config configs/c2i/nit_xl_pack_merge_radio_16384.yaml \
  --ckpt checkpoints/model_1000K.safetensors \
  --sample-dir ./samples \
  --height 512 \
  --width 512 \
  --per-proc-batch-size 32 \
  --mode sde \
  --num-steps 250 \
  --cfg-scale 2.05 \
  --guidance-low 0.0 \
  --guidance-high 0.7 \
  --slice_vae \