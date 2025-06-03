torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  projects/sample/sample_c2i_ddp.py \
  --config configs/c2i/nit_xl_pack_merge_radio_16384.yaml \
  --ckpt checkpoints/nit_xl_model_1000K.safetensors \
  --sample-dir ./samples \
  --height 256 \
  --width 256 \
  --per-proc-batch-size 32 \
  --mode sde \
  --num-steps 250 \
  --cfg-scale 2.25 \
  --guidance-low 0.0 \
  --guidance-high 0.7 \
  --slice_vae \