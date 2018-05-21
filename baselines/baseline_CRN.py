
# Copyright 2017 Xintong Han. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Baseline Cascaded refinement network. Training
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import json
import math
import os
import time

from utils import *

import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern",
                       "../../prepare_data/tfrecord/zalando-tps-points-new-train-?????-of-00032",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("train_dir", "tmp/",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("mode", "train", "Training or testing")
tf.flags.DEFINE_string("checkpoint", "", "Checkpoint path to resume training.")
tf.flags.DEFINE_string("output_dir", "model/results",
                       "Output directory of images.")
tf.flags.DEFINE_string("vgg_model_path", "../imagenet-vgg-verydeep-19.mat",
                       "model of the trained vgg net.")
tf.flags.DEFINE_string("stage", "", "stage of the training")


tf.flags.DEFINE_integer("number_of_steps", 1000000,
                        "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 10,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("batch_size", 4, "Size of mini batch.")
tf.flags.DEFINE_integer("num_preprocess_threads", 1, "")
tf.flags.DEFINE_integer("values_per_input_shard", 443, "")
tf.flags.DEFINE_integer("ngf", 64,
                        "number of generator filters in first conv layer")
tf.flags.DEFINE_integer("ndf", 64,
                        "number of discriminator filters in first conv layer")
# Summary
tf.flags.DEFINE_integer("summary_freq", 50, #100
                        "update summaries every summary_freq steps")
tf.flags.DEFINE_integer("progress_freq", 50, #100
                        "display progress every progress_freq steps")
tf.flags.DEFINE_integer("trace_freq", 0,
                        "trace execution every trace_freq steps")
tf.flags.DEFINE_integer("display_freq", 50, #300
                        "write current training images every display_freq steps")
tf.flags.DEFINE_integer("save_freq", 3000,
                        "save model every save_freq steps, 0 to disable")


tf.flags.DEFINE_float("mask_offset", 1.0, "Weight mask is emphasized.")
tf.flags.DEFINE_float("number_of_samples", 14221.0, "Samples in training set.")
tf.flags.DEFINE_float("lr", 0.0001, "Initial learning rate.")
tf.flags.DEFINE_float("beta1", 0.5, "momentum term of adam")
tf.flags.DEFINE_float("mask_l1_weight", 1.0, "Weight on L1 term of product mask.")
tf.flags.DEFINE_float("content_l1_weight", 1.0, "Weight on L1 term of content.")
tf.flags.DEFINE_float("perceptual_weight", 1.0, "weight on GAN term.")


tf.logging.set_verbosity(tf.logging.INFO)


Model = collections.namedtuple("Model",
                               "image_outputs,"
                               "gen_loss_GAN,p0,p1,p2,p3,p4,p5,"
                               "train, global_step")


def is_training():
  return FLAGS.mode == "train"


def create_generator(product_image, body_seg, skin_seg,
                     pose_map, generator_outputs_channels):
  """ Generator from product images, segs, poses to a segment map"""
  # Build inputs
  n_filters = [1024, 512, 512, 128, 32] # v1
  # n_filters = [1024, 512,512, 128, 128] # v2
  generator_input = tf.concat([product_image, body_seg, skin_seg, pose_map],
                               axis=-1)

  downsampled = tf.image.resize_area(generator_input,
                                     (16, 12),
                                     align_corners=False)
  net = slim.conv2d(downsampled, n_filters[0], [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_16_conv1')
  net = slim.conv2d(net, n_filters[0], [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_16_conv2')


  downsampled = tf.image.resize_area(generator_input,
                                     (32, 24),
                                     align_corners=False)
  upsampled_net = tf.image.resize_bilinear(net, (32, 24), align_corners=True)
  input = tf.concat([upsampled_net, downsampled], 3)
  net = slim.conv2d(input, n_filters[1], [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_32_conv1')
  net = slim.conv2d(net, n_filters[1], [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_32_conv2')


  downsampled = tf.image.resize_area(generator_input,
                                     (64, 48),
                                     align_corners=False)
  upsampled_net = tf.image.resize_bilinear(net, (64, 48), align_corners=True)
  input = tf.concat([upsampled_net, downsampled], 3)
  net = slim.conv2d(input, n_filters[2], [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_64_conv1')
  net = slim.conv2d(net, n_filters[2], [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_64_conv2')



  downsampled = tf.image.resize_area(generator_input,
                                     (128, 96),
                                     align_corners=False)
  upsampled_net = tf.image.resize_bilinear(net, (128, 96), align_corners=True)
  input = tf.concat([upsampled_net, downsampled], 3)
  net = slim.conv2d(input, n_filters[3], [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_128_conv1')
  net = slim.conv2d(net, n_filters[3], [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_128_conv2')


  downsampled = tf.image.resize_area(generator_input,
                                     (256, 192),
                                     align_corners=False)
  upsampled_net = tf.image.resize_bilinear(net, (256, 192), align_corners=True)
  input = tf.concat([upsampled_net, downsampled], 3)
  net = slim.conv2d(input, n_filters[4], [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_256_conv1')
  net = slim.conv2d(net, n_filters[4], [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_256_conv2')

  # finally output RGB image
  net = slim.conv2d(net, 3, [1, 1], rate=1,
                    activation_fn=None, scope='g_256_final')
  net = tf.tanh(net)
  return net  

def create_model(product_image, body_seg, skin_seg, pose_map, prod_seg, image):
  """Build the model given product image, skin/body segments, pose
     predict the product segmentation.
  """

  with tf.variable_scope("generator") as scope:
    out_channels = int(image.get_shape()[-1])
    image_outputs = create_generator(product_image, body_seg, skin_seg,
                               pose_map, out_channels)

  with tf.name_scope("generator_loss"):
    
    with tf.variable_scope("vgg_19"):
      vgg_real = build_vgg19(image, FLAGS.vgg_model_path)
      vgg_fake = build_vgg19(image_outputs, FLAGS.vgg_model_path, reuse=True)
      p0 = compute_error(vgg_real['input'],
                         vgg_fake['input']) # 256*256*3
      p1 = compute_error(vgg_real['conv1_2'],
                         vgg_fake['conv1_2']) / 1.6  # 128*128*64
      p2 = compute_error(vgg_real['conv2_2'],
                         vgg_fake['conv2_2']) / 2.3 # 64*64*128
      p3 = compute_error(vgg_real['conv3_2'],
                         vgg_fake['conv3_2']) / 1.8 # 32*32*256
      p4 = compute_error(vgg_real['conv4_2'],
                         vgg_fake['conv4_2']) / 3.5 # 16*16*512
      p5 = compute_error(vgg_real['conv5_2'],
                         vgg_fake['conv5_2']) * 8.5  # 8*8*512
      gen_loss = (p0 + p1 + p2 + p3 + p4 + p5) / 6.0 / 128.0 # 128 for normalize to [0.1]
  
  
  with tf.name_scope("generator_train"):
    # with tf.control_dependencies([discrim_train]):
    gen_tvars = [var for var in tf.trainable_variables()
                 if var.name.startswith("generator")]
    print(gen_tvars)
    gen_optim = tf.train.AdamOptimizer(FLAGS.lr)
    gen_train = gen_optim.minimize(gen_loss, var_list=gen_tvars)

  global_step = tf.contrib.framework.get_or_create_global_step()
  incr_global_step = tf.assign(global_step, global_step+1)

  return Model(
      gen_loss_GAN=gen_loss,
      p0=p0,
      p1=p1,
      p2=p2,
      p3=p3,
      p4=p4,
      p5=p5,
      image_outputs=image_outputs,
      train=tf.group(incr_global_step, gen_train),
      global_step=global_step)


def build_input():
  # Load input data
  input_queue = prefetch_input_data(
      tf.TFRecordReader(),
      FLAGS.input_file_pattern,
      is_training=is_training(),
      batch_size=FLAGS.batch_size,
      values_per_shard=FLAGS.values_per_input_shard,
      input_queue_capacity_factor=2,
      num_reader_threads=FLAGS.num_preprocess_threads)

  # Image processing and random distortion. Split across multiple threads
  # with each thread applying a slightly different distortion.
  # assert self.config.num_preprocess_threads % 2 == 0
  images_and_maps = []

  for thread_id in range(FLAGS.num_preprocess_threads):
    serialized_example = input_queue.dequeue()
    (encoded_image, encoded_prod_image, body_segment, prod_segment,
     skin_segment, pose_map, image_id) = parse_tf_example(serialized_example,
                                                          stage=FLAGS.stage)

    (image, product_image, body_segment, prod_segment,
     skin_segment, pose_map) = process_image(encoded_image,
                                             encoded_prod_image,
                                             body_segment,
                                             prod_segment,
                                             skin_segment,
                                             pose_map,
                                             is_training())

    images_and_maps.append([image, product_image, body_segment,
                            prod_segment, skin_segment, pose_map, image_id])

  # Batch inputs.
  queue_capacity = (7 * FLAGS.num_preprocess_threads *
                    FLAGS.batch_size)

  return tf.train.batch_join(images_and_maps,
                             batch_size=FLAGS.batch_size,
                             capacity=queue_capacity,
                             name="batch")





def deprocess_image(image, mask01=False):
  if not mask01:
    image = image / 2.0 + 0.5
  return tf.image.convert_image_dtype(image, dtype=tf.uint8)


def main(unused_argv):
  (image, product_image, body_segment, prod_segment, skin_segment,
   pose_map, image_id) = build_input()

  # Build model and loss function
  model = create_model(product_image, body_segment, skin_segment,
                       pose_map, prod_segment, image)

  # Summaries.
  with tf.name_scope("encode_images"):
    display_fetches = {
        "paths": image_id,
        "image": tf.map_fn(tf.image.encode_png, deprocess_image(image),
                           dtype=tf.string, name="image_pngs"),
        "product_image": tf.map_fn(tf.image.encode_png,
                                   deprocess_image(product_image),
                                   dtype=tf.string, name="prod_image_pngs"),
        "body_segment": tf.map_fn(tf.image.encode_png,
                                  deprocess_image(body_segment, True),
                                  dtype=tf.string, name="body_segment_pngs"),
        "skin_segment": tf.map_fn(tf.image.encode_png,
                                  deprocess_image(skin_segment),
                                  dtype=tf.string, name="skin_segment_pngs"),
        "prod_segment": tf.map_fn(tf.image.encode_png,
                                  deprocess_image(prod_segment, True),
                                  dtype=tf.string, name="prod_segment_pngs"),
        "image_outputs": tf.map_fn(tf.image.encode_png,
                             deprocess_image(model.image_outputs),
                             dtype=tf.string, name="image_output_pngs"),

    }


    test_fetches = {"image_outputs": tf.map_fn(tf.image.encode_png,
               deprocess_image(model.image_outputs),
               dtype=tf.string, name="image_output_pngs"),
               "paths": image_id,}

  tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)

  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/values", var)

  with tf.name_scope("parameter_count"):
    parameter_count = tf.reduce_sum(
        [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

  saver = tf.train.Saver(max_to_keep=100)
  sv = tf.train.Supervisor(logdir=FLAGS.output_dir,
                           save_summaries_secs=0, saver=None)
  with sv.managed_session() as sess:
    tf.logging.info("parameter_count = %d" % sess.run(parameter_count))
    
    if FLAGS.checkpoint != "":
      tf.logging.info("loading model from checkpoint")
      checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
      if checkpoint == None:
        checkpoint = FLAGS.checkpoint
      saver.restore(sess, checkpoint)

    if FLAGS.mode == "test":
      # testing
      # at most, process the test data once
      tf.logging.info("test!")
      with open(os.path.join(FLAGS.output_dir, "options.json"), "a") as f:
        f.write(json.dumps(vars(FLAGS), sort_keys=True, indent=4))

      start = time.time()
      max_steps = FLAGS.number_of_steps
      for step in range(max_steps):
        def should(freq):
          return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

        results = sess.run(test_fetches)

        image_dir = os.path.join(FLAGS.output_dir, "images")
        if not os.path.exists(image_dir):
          os.makedirs(image_dir)

        for i, in_path in enumerate(results["paths"]):
          name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
          filename = name + ".png"
          out_path = os.path.join(image_dir, filename)
          contents = results["image_outputs"][i]
          with open(out_path, "wb") as f:
            f.write(contents)

    else:
      # training
      with open(os.path.join(FLAGS.output_dir, "options.json"), "a") as f:
        f.write(json.dumps(vars(FLAGS), sort_keys=True, indent=4))

      start = time.time()
      max_steps = FLAGS.number_of_steps
      for step in range(max_steps):
        def should(freq):
          return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

        options = None
        run_metadata = None
        if should(FLAGS.trace_freq):
          options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()

        fetches = {
            "train": model.train,
            "global_step": sv.global_step,
        }

        if should(FLAGS.progress_freq):
          fetches["gen_loss_GAN"] = model.gen_loss_GAN
          fetches["p0"] = model.p0
          fetches["p1"] = model.p1
          fetches["p2"] = model.p2
          fetches["p3"] = model.p3
          fetches["p4"] = model.p4
          fetches["p5"] = model.p5

        if should(FLAGS.summary_freq):
          fetches["summary"] = sv.summary_op

        if should(FLAGS.display_freq):
          fetches["display"] = display_fetches

        results = sess.run(fetches, options=options, run_metadata=run_metadata)

        if should(FLAGS.summary_freq):
          tf.logging.info("recording summary")
          sv.summary_writer.add_summary(
              results["summary"], results["global_step"])

        if should(FLAGS.display_freq):
          tf.logging.info("saving display images")
          filesets = save_images(results["display"],
                                 image_dict=["body_segment", "skin_segment",
                                             "prod_segment",
                                             "product_image", "image", 
                                             "image_outputs"],
                                 output_dir=FLAGS.output_dir,
                                 step=results["global_step"])
          append_index(filesets, 
                       image_dict=["body_segment", "skin_segment",
                                   "prod_segment",
                                   "product_image", "image", 
                                   "image_outputs"],
                       output_dir=FLAGS.output_dir,
                       step=True)

        if should(FLAGS.trace_freq):
          tf.logging.info("recording trace")
          sv.summary_writer.add_run_metadata(
              run_metadata, "step_%d" % results["global_step"])

        if should(FLAGS.progress_freq):
          # global_step will have the correct step count if we resume from a
          # checkpoint
          train_epoch = math.ceil(
              results["global_step"] / FLAGS.number_of_samples)
          rate = (step + 1) * FLAGS.batch_size / (time.time() - start)
          tf.logging.info("progress epoch %d step %d  image/sec %0.1f" %
                (train_epoch, results["global_step"], rate))
          tf.logging.info("gen_loss_GAN: %f" % results["gen_loss_GAN"])
          tf.logging.info("p0: %f" % results["p0"])
          tf.logging.info("p1: %f" % results["p1"])
          tf.logging.info("p2: %f" % results["p2"])
          tf.logging.info("p3: %f" % results["p3"])
          tf.logging.info("p4: %f" % results["p4"])
          tf.logging.info("p5: %f" % results["p5"])

        if should(FLAGS.save_freq):
          tf.logging.info("saving model")
          saver.save(sess, os.path.join(FLAGS.output_dir, "model"),
                     global_step=sv.global_step)

        if sv.should_stop():
          break


if __name__ == "__main__":
  tf.app.run()
