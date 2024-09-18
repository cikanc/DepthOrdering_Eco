export CUDA_VISIBLE_DEVICES=0

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python infer2.py \
    --img_path /tmp/EcoDepth/train \
    --max_depth 80.0 \
    --min_depth 1e-3 \
    --ckpt_dir /tmp/EcoDepth/depth/checkpoints/nyu.ckpt
