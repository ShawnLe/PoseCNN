#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/linemod_driller_det_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

time ./tools/train_net.py --gpu 0 \
  --network vgg16_det \
  --weights data/imagenet_models/vgg16.npy \
  --imdb linemod_driller_train \
  --cfg experiments/cfgs/linemod_driller_det.yml \
  --iters 160000
