#!/bin/bash

PROJ_DIR="../"

TRAIN_LABEL_FILE=${PROJ_DIR}"data/viton_train_images.txt"
IMAGE_DIR=${PROJ_DIR}"data/women_top/"
POSE_DIR=${PROJ_DIR}"data/pose.pkl"
SEG_DIR=${PROJ_DIR}"data/segment/"
OUTPUT_DIR=${PROJ_DIR}"prepare_data/tfrecord/"
TPS_DIR=${PROJ_DIR}"data/tps/"

python build_viton.py  --image_dir="${IMAGE_DIR}" \
  --pose_dir="${POSE_DIR}" \
  --segment_dir="${SEG_DIR}" \
  --train_label_file="${TRAIN_LABEL_FILE}" \
  --output_dir="${OUTPUT_DIR}" \
  --tps_dir="${TPS_DIR}" \
  --prefix="zalando-"


# segmentation mapping
# 1 Hat, 2 Hair, 4 Sunglasses, 13 Face, 11 Scarf, 3 Glove, 18 Left-shoe, 19 Right-shoe
# 5 Upper-clothes, 6 Dress, 7 Coat, 10 Jumpsuits, 12 Skirt
# 8 Socks, 9 Pants, 14 Left-arm, 15 Right-arm, 16 Left-leg, 17 Right-leg
