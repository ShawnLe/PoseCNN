#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/lov_color_3d.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
#export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
#time ./tools/train_net.py --gpu 0 \
#  --network vgg16_convs \
#  --weights data/imagenet_models/vgg16_convs.npy \
#  --imdb lov_train \
#  --cfg experiments/cfgs/lov_color_3d.yml \
#  --iters 80000

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output/lov/lov_train/vgg16_fcn_color_single_frame_3d_lov_iter_80000.ckpt \
  --imdb lov_keyframe \
  --cfg experiments/cfgs/lov_color_3d.yml \
  --rig data/LOV/camera.json \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --background data/cache/lov_train_backgrounds.pkl
