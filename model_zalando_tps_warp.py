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

""" Stage2: given control point generate warpped images and use it for refinement.
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

from tps_transformer import tps_stn

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern",
                       "./prepare_data/tfrecord/zalando-train-?????-of-00032",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("mode", "train", "Training or testing")
tf.flags.DEFINE_string("checkpoint", "", "Checkpoint path to resume training.")
tf.flags.DEFINE_string("gen_checkpoint", "",
                       "Checkpoint path to the initial generative model.")
tf.flags.DEFINE_string("output_dir", "model/stage2/",
                       "Output directory of images.")
tf.flags.DEFINE_string("vgg_model_path", "./model/imagenet-vgg-verydeep-19.mat",
                       "model of the trained vgg net.")

tf.flags.DEFINE_integer("number_of_steps", 100000,
                        "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 10,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("batch_size", 16, "Size of mini batch.")
tf.flags.DEFINE_integer("num_preprocess_threads", 1, "")
tf.flags.DEFINE_integer("values_per_input_shard", 433, "")
tf.flags.DEFINE_integer("ngf", 64,
                        "number of generator filters in first conv layer")
tf.flags.DEFINE_integer("ndf", 64,
                        "number of discriminator filters in first conv layer")
# Summary
tf.flags.DEFINE_integer("summary_freq", 50, #100
                        "update summaries every summary_freq steps")
tf.flags.DEFINE_integer("progress_freq", 10, #100
                        "display progress every progress_freq steps")
tf.flags.DEFINE_integer("trace_freq", 0,
                        "trace execution every trace_freq steps")
tf.flags.DEFINE_integer("display_freq", 300, #300
                        "write current training images every display_freq steps")
tf.flags.DEFINE_integer("save_freq", 1000,
                        "save model every save_freq steps, 0 to disable")

tf.flags.DEFINE_float("number_of_samples", 14221.0, "Samples in training set.")
tf.flags.DEFINE_float("lr", 0.0002, "Initial learning rate.")
tf.flags.DEFINE_float("beta1", 0.5, "momentum term of adam")
tf.flags.DEFINE_float("content_l1_weight", 0.2, "Weight on L1 term of content.")
tf.flags.DEFINE_float("perceptual_weight", 0.8, "weight on GAN term.")
tf.flags.DEFINE_float("tv_weight", 0.000005, "weight on TV term.")
tf.flags.DEFINE_float("mask_weight", 0.1, "weight on the selection mask.")


tf.logging.set_verbosity(tf.logging.INFO)

Model = collections.namedtuple("Model",
                               "gen_image_outputs, stn_image_outputs,"
                               "image_outputs, prod_mask_outputs,"
                               "mask_loss, tv_loss,"
                               "gen_loss_GAN, select_mask,"
                               "gen_loss_content_L1, perceptual_loss,"
                               "train, global_step")

FINAL_HEIGHT = 256
FINAL_WIDTH = 192

def is_training():
  return FLAGS.mode == "train"



def create_generator(product_image, body_seg, skin_seg,
                     pose_map, generator_outputs_channels):
  """ Generator from product images, segs, poses to a segment map"""
  # Build inputs
  generator_inputs = tf.concat([product_image, body_seg, skin_seg, pose_map],
                               axis=-1)
  # generator_inputs = tf.concat([body_seg, skin_seg, pose_map],
  #                              axis=-1)

  layers = []

  # encoder_1: [batch, 256, 192, in_channels] => [batch, 128, 96, ngf]
  with tf.variable_scope("encoder_1"):
    output = conv(generator_inputs, FLAGS.ngf, stride=2)
    layers.append(output)

  layer_specs = [
      # encoder_2: [batch, 128, 96, ngf] => [batch, 64, 48, ngf * 2]
      FLAGS.ngf * 2,
      # encoder_3: [batch, 64, 48, ngf * 2] => [batch, 32, 24, ngf * 4]
      FLAGS.ngf * 4,
      # encoder_4: [batch, 32, 24, ngf * 4] => [batch, 16, 12, ngf * 8]
      FLAGS.ngf * 8,
      # encoder_5: [batch, 16, 12, ngf * 8] => [batch, 8, 6, ngf * 8]
      FLAGS.ngf * 8,
      # encoder_6: [batch, 8, 6, ngf * 8] => [batch, 4, 3, ngf * 8]
      FLAGS.ngf * 8,
      # encoder_7: [batch, 4, 3, ngf * 8] => [batch, 2, 1, ngf * 8]
      # FLAGS.ngf * 8,
  ]

  for out_channels in layer_specs:
    with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
      rectified = lrelu(layers[-1], 0.2)
      # [batch, in_height, in_width, in_channels]
      # => [batch, in_height/2, in_width/2, out_channels]
      convolved = conv(rectified, out_channels, stride=2)
      output = batch_norm(convolved, is_training())
      layers.append(output)


  layer_specs = [
      # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
      # (FLAGS.ngf * 8, 0.5),
      # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
      # (FLAGS.ngf * 8, 0.5),
      # decoder_6: [batch, 4, 3, ngf * 8 * 2] => [batch, 8, 6, ngf * 8 * 2]
      (FLAGS.ngf * 8, 0.5),
      # decoder_5: [batch, 8, 12, ngf * 8 * 2] => [batch, 16, 12, ngf * 8 * 2]
      (FLAGS.ngf * 8, 0.0),
      # decoder_4: [batch, 16, 12, ngf * 8 * 2] => [batch, 32, 24, ngf * 4 * 2]
      (FLAGS.ngf * 4, 0.0),
      # decoder_3: [batch, 32, 24, ngf * 4 * 2] => [batch, 64, 48, ngf * 2 * 2]
      (FLAGS.ngf * 2, 0.0),
      # decoder_2: [batch, 64, 48, ngf * 2 * 2] => [batch, 128, 96, ngf * 2]
      (FLAGS.ngf, 0.0),
  ]

  num_encoder_layers = len(layers)
  for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
    skip_layer = num_encoder_layers - decoder_layer - 1
    with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
      if decoder_layer == 0:
        # first decoder layer doesn't have skip connections
        # since it is directly connected to the skip_layer
        input = layers[-1]
      else:
        input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

      rectified = tf.nn.relu(input)
      # [batch, in_height, in_width, in_channels]
      # => [batch, in_height*2, in_width*2, out_channels]
      output = deconv(rectified, out_channels)
      output = batch_norm(output, is_training())

      if dropout > 0.0 and is_training():
        output = tf.nn.dropout(output, keep_prob=1 - dropout)

      layers.append(output)

  # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256,
  # generator_outputs_channels]
  with tf.variable_scope("decoder_1"):
    input = tf.concat([layers[-1], layers[0]], axis=3)
    rectified = tf.nn.relu(input)
    output = deconv(rectified, generator_outputs_channels)
    output = tf.tanh(output)
    layers.append(output)

  return layers[-1]


def create_refine_generator(stn_image_outputs, gen_image_outputs):
  generator_input = tf.concat([stn_image_outputs, gen_image_outputs],
                               axis=-1)

  downsampled = tf.image.resize_area(generator_input,
                                     (256, 192),
                                     align_corners=False)
  net = slim.conv2d(downsampled, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_256_conv1')
  net = slim.conv2d(net, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_256_conv2')
  net = slim.conv2d(net, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_256_conv3')

  # output a selection mask:
  net = slim.conv2d(net, 1, [1, 1], rate=1,
                    activation_fn=None, scope='g_1024_final')
  net = tf.sigmoid(net)
  return net  



def extract_product_fg(product_image):
  image = (product_image / 2.0 + 0.5) * 255.0
  prod_image_fg = tf.less(tf.reduce_mean(image, axis=-1), 253.9)
  return tf.cast(tf.expand_dims(prod_image_fg, axis=-1), tf.float32)



def create_model(product_image,
                 body_seg, skin_seg, pose_map, prod_seg,
                 image, tps_points):
  """Build the model given product image, skin/body segments, pose
     predict the product segmentation.
  """

  with tf.variable_scope("generator") as scope:
  # downsample image and prod_image and input them into the coarse model
    downsample_prod_image = tf.image.resize_images(product_image,
                                      size=[256, 192],
                                      method=tf.image.ResizeMethod.BILINEAR)
    out_channels = int(prod_seg.get_shape()[-1] + image.get_shape()[-1])
    gen_image_outputs = create_generator(downsample_prod_image, body_seg, skin_seg,
                                         pose_map, out_channels)
    gen_image_outputs = gen_image_outputs[:,:,:,prod_seg.get_shape()[-1]:]
  
    gen_image_outputs = tf.image.resize_area(gen_image_outputs,
                                             (FINAL_HEIGHT, FINAL_WIDTH),
                                             align_corners=False)
  with tf.variable_scope("stn_generator") as scope:
    prod_image_fg = extract_product_fg(product_image)
    
    stn_outputs = tps_stn(tf.concat([prod_image_fg, product_image], axis=-1),
                                    10, 10,
                                    tps_points,
                                    product_image.get_shape()[1:3])


    prod_mask_outputs = stn_outputs[:,:,:,:1]
    stn_image_outputs = stn_outputs[:,:,:,1:]
  with tf.variable_scope("refine_generator") as scope:
    select_mask = create_refine_generator(stn_image_outputs,
                                          gen_image_outputs)
    # only look at the prod_seg region in select_mask
    select_mask = prod_seg * select_mask
    image_outputs = select_mask * stn_image_outputs + (1 - select_mask) * gen_image_outputs
  with tf.name_scope("generator_loss"):
    gen_loss_content_L1 = tf.reduce_mean(tf.abs(image - image_outputs))
    with tf.variable_scope("vgg_19"):
      vgg_real = build_vgg19(image, FLAGS.vgg_model_path)
      vgg_fake = build_vgg19(image_outputs, FLAGS.vgg_model_path, reuse=True)
      p1 = compute_error(vgg_real['conv1_2'],
                         vgg_fake['conv1_2']) / 5.3 * 2.5  # 128*128*64
      p2 = compute_error(vgg_real['conv2_2'],
                         vgg_fake['conv2_2']) / 2.7  / 1.2 # 64*64*128
      p3 = compute_error(vgg_real['conv3_2'],
                         vgg_fake['conv3_2']) / 1.35 / 2.3 # 32*32*256
      p4 = compute_error(vgg_real['conv4_2'],
                         vgg_fake['conv4_2']) / 0.67 / 8.2 # 16*16*512
      p5 = compute_error(vgg_real['conv5_2'],
                         vgg_fake['conv5_2']) / 0.16  # 8*8*512
      perceptual_loss = (p3 + p4 + p5) / 3.0 / 128.0 # 128 for normalize to [0.1]
    
    mask_loss = tf.reduce_mean(select_mask)
    tv_loss = tf.reduce_mean(tf.image.total_variation(select_mask))

    gen_loss = (FLAGS.content_l1_weight * gen_loss_content_L1 +
                FLAGS.perceptual_weight * perceptual_loss - 
                FLAGS.mask_weight * mask_loss +
                FLAGS.tv_weight *  tv_loss # TV loss
                )



  with tf.name_scope("generator_train"):
    gen_tvars = [var for var in tf.trainable_variables()
                 if var.name.startswith("refine_generator")]
    gen_optim = tf.train.AdamOptimizer(FLAGS.lr, FLAGS.beta1)
    gen_train = gen_optim.minimize(gen_loss, var_list=gen_tvars)

  global_step = tf.contrib.framework.get_or_create_global_step()
  incr_global_step = tf.assign(global_step, global_step+1)

  return Model(
      gen_loss_GAN=gen_loss,
      gen_loss_content_L1=gen_loss_content_L1,
      perceptual_loss=perceptual_loss,
      mask_loss=mask_loss,
      tv_loss=tv_loss,
      gen_image_outputs=gen_image_outputs,
      stn_image_outputs=stn_image_outputs,
      select_mask=select_mask,
      image_outputs=image_outputs,
      prod_mask_outputs=prod_mask_outputs,
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
  images_and_maps = []

  for thread_id in range(FLAGS.num_preprocess_threads):
    serialized_example = input_queue.dequeue()
    (encoded_image, encoded_prod_image, body_segment, prod_segment,
     skin_segment, pose_map, image_id, tps_points) = parse_tf_example(serialized_example,
                                                                      "tps_points")

    (image, product_image, body_segment, prod_segment,
     skin_segment, pose_map) = process_image(encoded_image,
                                             encoded_prod_image,
                                             body_segment,
                                             prod_segment,
                                             skin_segment,
                                             pose_map,
                                             is_training(),
                                             height=FINAL_HEIGHT,
                                             width=FINAL_WIDTH,
                                             different_image_size=True)

    images_and_maps.append([image, product_image, body_segment,
                            prod_segment, skin_segment, pose_map, image_id, tps_points])

  # Batch inputs.
  queue_capacity = (2 * FLAGS.num_preprocess_threads *
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
  # Fetch input
  (image, product_image, body_segment, prod_segment, skin_segment,
   pose_map, image_id, tps_points) = build_input()


  
  # Build model and loss function
  model = create_model(product_image,
                       body_segment, skin_segment,
                       pose_map, prod_segment,
                       image, tps_points)

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
                                  deprocess_image(skin_segment, True),
                                  dtype=tf.string, name="skin_segment_pngs"),
        "prod_segment": tf.map_fn(tf.image.encode_png,
                                  deprocess_image(prod_segment, True),
                                  dtype=tf.string, name="prod_segment_pngs"),
        "prod_mask_outputs": tf.map_fn(tf.image.encode_png,
                             deprocess_image(model.prod_mask_outputs, True),
                             dtype=tf.string, name="mask_output_pngs"),
        "stn_image_outputs": tf.map_fn(tf.image.encode_png,
                             deprocess_image(model.stn_image_outputs),
                             dtype=tf.string, name="stn_image_output_pngs"),
        "gen_image_outputs": tf.map_fn(tf.image.encode_png,
                             deprocess_image(model.gen_image_outputs),
                             dtype=tf.string, name="gen_image_output_pngs"),
        "image_outputs": tf.map_fn(tf.image.encode_png,
                             deprocess_image(model.image_outputs),
                             dtype=tf.string, name="image_output_pngs"),
        "select_mask": tf.map_fn(tf.image.encode_png,
                                 deprocess_image(model.select_mask, True),
                             dtype=tf.string, name="image_output_pngs"),

    }


    test_fetches = {"image_outputs": tf.map_fn(tf.image.encode_png,
               deprocess_image(model.image_outputs),
               dtype=tf.string, name="image_output_pngs"),
               "paths": image_id,}

  tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
  tf.summary.scalar("generator_loss_content_L1", model.gen_loss_content_L1)
  tf.summary.scalar("perceptual_loss", model.perceptual_loss)

  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/values", var)

  with tf.name_scope("parameter_count"):
    parameter_count = tf.reduce_sum(
        [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

  saver = tf.train.Saver(max_to_keep=100)

  if FLAGS.checkpoint != "":
    restore_fn = None
  else:
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.gen_checkpoint)
    if checkpoint_file == None:
      checkpoint_file = FLAGS.gen_checkpoint
    saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith("generator")])
    def restore_fn(sess):
      return saver.restore(sess, checkpoint_file)

  saver2 = tf.train.Saver(max_to_keep=100)

  sv = tf.train.Supervisor(logdir=FLAGS.output_dir,
                           save_summaries_secs=0, saver=None,
                           init_fn=restore_fn)

  with sv.managed_session() as sess:
    tf.logging.info("parameter_count = %d" % sess.run(parameter_count))
    
    if FLAGS.checkpoint != "":
      tf.logging.info("loading model from checkpoint")
      checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
      if checkpoint == None:
        checkpoint = FLAGS.checkpoint
      tf.logging.info(checkpoint)
      saver2.restore(sess, checkpoint)

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
          fetches["gen_loss_content_L1"] = model.gen_loss_content_L1
          fetches["gen_loss_GAN"] = model.gen_loss_GAN
          fetches["perceptual_loss"] = model.perceptual_loss
          fetches["tv_loss"] = model.tv_loss
          fetches["mask_loss"] = model.mask_loss

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
                                             "prod_segment", "prod_mask_outputs",
                                             "product_image", "stn_image_outputs",
                                             "gen_image_outputs", "select_mask",
                                             "image_outputs", "image"],
                                 output_dir=FLAGS.output_dir,
                                 step=results["global_step"])

          append_index(filesets, 
                       image_dict=["body_segment", "skin_segment",
                                   "prod_segment", "prod_mask_outputs",
                                   "product_image", "stn_image_outputs",
                                   "gen_image_outputs", "select_mask",
                                   "image_outputs", "image"],
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
          tf.logging.info("gen_loss_content_L1: %f" % results["gen_loss_content_L1"])
          tf.logging.info("perceptual_loss: %f" % results["perceptual_loss"])
          tf.logging.info("mask_loss: %f" % results["mask_loss"])
          tf.logging.info("tv_loss: %f" % results["tv_loss"])

        if should(FLAGS.save_freq):
          tf.logging.info("saving model")
          saver2.save(sess, os.path.join(FLAGS.output_dir, "model"),
                     global_step=sv.global_step)

        if sv.should_stop():
          break


if __name__ == "__main__":
  tf.app.run()
