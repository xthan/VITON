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

""" CAGAN
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

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern",
                       "../../prepare_data/tfrecord/zalando-tps-points-new-train-?????-of-00032",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("train_dir", "tmp/",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("mode", "train", "Training or testing")
tf.flags.DEFINE_string("checkpoint", "", "Checkpoint path to resume training.")
tf.flags.DEFINE_string("output_dir", "tmp",
                       "Output directory of images.")

tf.flags.DEFINE_integer("number_of_steps", 1000000,
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
tf.flags.DEFINE_integer("summary_freq", 100,
                        "update summaries every summary_freq steps")
tf.flags.DEFINE_integer("progress_freq", 20,
                        "display progress every progress_freq steps")
tf.flags.DEFINE_integer("trace_freq", 0,
                        "trace execution every trace_freq steps")
tf.flags.DEFINE_integer("display_freq", 100,
                        "write current training images every display_freq steps")
tf.flags.DEFINE_integer("save_freq", 1000,
                        "save model every save_freq steps, 0 to disable")


tf.flags.DEFINE_float("number_of_samples", 14221.0, "Samples in training set.")
tf.flags.DEFINE_float("lr", 0.0002, "Initial learning rate.")
tf.flags.DEFINE_float("beta1", 0.5, "momentum term of adam")
tf.flags.DEFINE_float("content_l1_weight", 1.0, "Weight on L1 term of content.")
tf.flags.DEFINE_float("gan_weight", 1.0, "weight on GAN term.")
tf.flags.DEFINE_float("id_weight", .1, "weight L1 mask (id).")
tf.flags.DEFINE_float("cycle_weight", 1.0, "weight of CycleGAN loss.")


tf.logging.set_verbosity(tf.logging.INFO)

EPS = 1e-12


Model = collections.namedtuple("Model",
                               "image_outputs,mask_outputs,cycle_outputs,"
                               "gen_loss_GAN, gen_loss_id, cycle_loss,"
                               "gen_loss, discrim_loss,"
                               "train, global_step")



def is_training():
  return FLAGS.mode == "train"


def create_generator(image1, product_image1, product_image2, generator_outputs_channels):
  """ Generator from product images, coditions to output."""
  # Build inputs
  generator_inputs = tf.concat([image1, product_image1, product_image2],
                               axis=-1)
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
      # FLAGS.ngf * 8,
      # encoder_7: [batch, 4, 3, ngf * 8] => [batch, 2, 1, ngf * 8]
      # FLAGS.ngf * 8,
  ]

  for out_channels in layer_specs:
    with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
      rectified = lrelu(layers[-1], 0.2)
      # [batch, in_height, in_width, in_channels]
      # => [batch, in_height/2, in_width/2, out_channels]
      convolved = conv(rectified, out_channels, stride=2)
      output = batchnorm(convolved)
      layers.append(output)

  layer_specs = [
      # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
      # (FLAGS.ngf * 8, 0.5),
      # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
      # (FLAGS.ngf * 8, 0.5),
      # decoder_6: [batch, 4, 3, ngf * 8 * 2] => [batch, 8, 6, ngf * 8 * 2]
      # (FLAGS.ngf * 8, 0.5),
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
    mask_output = tf.sigmoid(output[:,:,:,3:])
    image_output = tf.tanh(output[:,:,:,:3])
    
  return image_output, mask_output


# Image GAN discriminator
def create_discriminator(discrim_inputs, discrim_targets):
  n_layers = 3
  layers = []
  # 2x [batch, height, width, in_channels] => [batch, height, width,
  # in_channels * 2]
  input = tf.concat([discrim_inputs, discrim_targets], axis=3)

  # layer_1: [batch, 256, 192, in_channels * 2] => [batch, 128, 96, ndf]
  with tf.variable_scope("layer_1"):
    convolved = conv(input, FLAGS.ndf, stride=2)
    rectified = lrelu(convolved, 0.2)
    layers.append(rectified)

  # layer_2: [batch, 128, 96, ndf] => [batch, 64, 48, ndf * 2]
  # layer_3: [batch, 64, 48, ndf * 2] => [batch, 32, 24, ndf * 4]
  # layer_4: [batch, 32, 24, ndf * 4] => [batch, 16, 12, ndf * 8]
  # layer_5: [batch, 16, 12, ndf * 8] => [batch, 8, 6, ndf * 8]
  stride = 2
  for i in range(n_layers):
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
      out_channels = FLAGS.ndf * min(2**(1+i), 8)
      # stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
      convolved = conv(layers[-1], out_channels, stride=stride)
      normalized = batch_norm(convolved, is_training())
      rectified = lrelu(normalized, 0.2)
      layers.append(rectified)

  # layer_7: [batch, 8, 6, ndf * 8] => [batch, 7, 5, ndf * 8]
  with tf.variable_scope("layer_%d" % (len(layers) + 1)):
    convolved = conv(rectified, out_channels=1, stride=1)
    output = tf.sigmoid(convolved)
    layers.append(output)

  return layers[-1]


def create_model(image, product_image1, product_image2):
  """Build the model given product image, skin/body segments, pose
     predict the product segmentation.
  """
  with tf.variable_scope("generator") as scope:
    image_outputs, mask_outputs = create_generator(image, product_image1, product_image2, 4)
    image_outputs = mask_outputs * image_outputs + (1 - mask_outputs) * image
  # create two copies of discriminator, one for real pairs and one for fake pairs
  # they share the same underlying variables
  with tf.name_scope("real_discriminator"):
    with tf.variable_scope("discriminator"):
      # 2x [batch, height, width, channels] => [batch, 30, 22, 1]
      predict_real = create_discriminator(product_image1, image)

  with tf.name_scope("fake_discriminator"):
    with tf.variable_scope("discriminator", reuse=True):
      # 2x [batch, height, width, channels] => [batch, 30, 22, 1]
      predict_fake = create_discriminator(product_image2, image_outputs)

  with tf.name_scope("unmatch_discriminator"):
    with tf.variable_scope("discriminator", reuse=True):
      # 2x [batch, height, width, channels] => [batch, 30, 22, 1]
      predict_unmatch = create_discriminator(product_image2, image)


  with tf.name_scope("discriminator_loss"):
    # minimizing -tf.log will try to get inputs to 1
    # predict_real => 1
    # predict_fake => 0
    discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) +
                                    tf.log(1 - predict_fake + EPS) + 
                                    tf.log(1 - predict_unmatch + EPS)
                                    ))


  with tf.name_scope("generator_loss"):
    # predict_fake => 1
    # abs(targets - outputs) => 0
    gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
    gen_loss_id = tf.reduce_mean(mask_outputs)
  
  with tf.name_scope("cycle_generator"):
    with tf.variable_scope("generator", reuse=True):
      cycle_outputs, _ = create_generator(image_outputs, product_image2, product_image1, 4)
      cycle_loss = tf.reduce_mean(tf.abs(image - cycle_outputs))
  

  gen_loss = (gen_loss_GAN * FLAGS.gan_weight + FLAGS.id_weight * gen_loss_id +
              FLAGS.cycle_weight * cycle_loss)
    
  with tf.name_scope("discriminator_train"):
    discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
    print(discrim_tvars)
    discrim_optim = tf.train.AdamOptimizer(FLAGS.lr, FLAGS.beta1)
    discrim_train = discrim_optim.minimize(discrim_loss, var_list=discrim_tvars)

  with tf.name_scope("generator_train"):
    with tf.control_dependencies([discrim_train]):
      gen_tvars = [var for var in tf.trainable_variables()
                   if var.name.startswith("generator")]
      gen_optim = tf.train.AdamOptimizer(FLAGS.lr, FLAGS.beta1)
      gen_train = gen_optim.minimize(gen_loss, var_list=gen_tvars)

  global_step = tf.contrib.framework.get_or_create_global_step()
  incr_global_step = tf.assign(global_step, global_step+1)

  return Model(
      gen_loss_GAN=gen_loss_GAN,
      gen_loss_id=gen_loss_id,
      gen_loss=gen_loss,
      discrim_loss=discrim_loss,
      cycle_loss=cycle_loss,
      mask_outputs=mask_outputs,
      image_outputs=image_outputs,
      cycle_outputs=cycle_outputs,
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
     skin_segment, pose_map, image_id) = parse_tf_example(serialized_example)

    (image, product_image, body_segment, prod_segment,
     skin_segment, pose_map) = process_image(encoded_image,
                                             encoded_prod_image,
                                             body_segment,
                                             prod_segment,
                                             skin_segment,
                                             pose_map,
                                             is_training())

    images_and_maps.append([image, product_image, image_id])

  # Batch inputs.
  queue_capacity = (2 * FLAGS.num_preprocess_threads *
                    FLAGS.batch_size)

  return tf.train.batch_join(images_and_maps,
                             batch_size=FLAGS.batch_size,
                             capacity=queue_capacity,
                             name="batch")



def deprocess_image(image, mask01=False):
  if not mask01:
    image = image / 2 + 0.5
  return tf.image.convert_image_dtype(image, dtype=tf.uint8)


def main(unused_argv):
  (image, product_image1, image_id) = build_input()
  # (_, product_image2, _) = build_input()
  product_image2 = tf.reverse(product_image1, axis=[0])
  print(product_image2)
  # Build model and loss function
  model = create_model(image, product_image1, product_image2)

  # Summaries.
  with tf.name_scope("encode_images"):
    display_fetches = {
        "paths": image_id,
        "image": tf.map_fn(tf.image.encode_png, deprocess_image(image),
                           dtype=tf.string, name="image_pngs"),
        "product_image1": tf.map_fn(tf.image.encode_png,
                                   deprocess_image(product_image1),
                                   dtype=tf.string, name="prod_image_pngs1"),
        "product_image2": tf.map_fn(tf.image.encode_png,
                                   deprocess_image(product_image2),
                                   dtype=tf.string, name="prod_image_pngs2"),
        "image_outputs": tf.map_fn(tf.image.encode_png,
                             deprocess_image(model.image_outputs),
                             dtype=tf.string, name="image_output_pngs"),
        "mask_outputs": tf.map_fn(tf.image.encode_png,
                             deprocess_image(model.mask_outputs, True),
                             dtype=tf.string, name="image_output_pngs"),
        "cycle_outputs": tf.map_fn(tf.image.encode_png,
                             deprocess_image(model.cycle_outputs),
                             dtype=tf.string, name="cycle_outputs_pngs"),

    }

  tf.summary.scalar("gen_loss_GAN", model.gen_loss_GAN)
  tf.summary.scalar("discrim_loss", model.discrim_loss)
  tf.summary.scalar("gen_loss_id", model.gen_loss_id)
  tf.summary.scalar("cycle_loss", model.cycle_loss)
  tf.summary.scalar("gen_loss", model.gen_loss)


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
    else:
      with open(os.path.join(FLAGS.output_dir, "options.json"), "a") as f:
        f.write(json.dumps(vars(FLAGS), sort_keys=True, indent=4))

      # training
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
          fetches["discrim_loss"] = model.discrim_loss
          fetches["gen_loss_id"] = model.gen_loss_id
          fetches["cycle_loss"] = model.cycle_loss
          fetches["gen_loss"] = model.gen_loss

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
                                 image_dict=["image", "product_image1",
                                             "product_image2", "image_outputs",
                                             "mask_outputs", "cycle_outputs", ],
                                 output_dir=FLAGS.output_dir,
                                 step=results["global_step"])
          append_index(filesets, 
                       image_dict=["image", "product_image1",
                                   "product_image2", "image_outputs",
                                   "mask_outputs", "cycle_outputs", ],
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
          tf.logging.info("discrim_loss: %f" % results["discrim_loss"])
          tf.logging.info("gen_loss_id: %f" % results["gen_loss_id"])
          tf.logging.info("cycle_loss: %f" % results["cycle_loss"])
          tf.logging.info("gen_loss: %f" % results["gen_loss"])

        if should(FLAGS.save_freq):
          tf.logging.info("saving model")
          saver.save(sess, os.path.join(FLAGS.output_dir, "model"),
                     global_step=sv.global_step)

        if sv.should_stop():
          break


if __name__ == "__main__":
  tf.app.run()
