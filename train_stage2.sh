python model_zalando_tps_warp.py \
  --output_dir model/stage2/ \
  --gen_checkpoint model/stage1/model-15000 \
  --input_file_pattern "./prepare_data/tfrecord/zalando-train-?????-of-00032"