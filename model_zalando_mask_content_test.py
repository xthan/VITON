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

""" Test for Stage 1: from product image + body segment +
    pose + face/hair predict a coarse result and product segment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

import numpy as np
import scipy.io as sio
import scipy.misc
import tensorflow as tf

from utils import *
from model_zalando_mask_content import create_model


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("pose_dir", "data/pose/",
                       "Directory containing poses.")
tf.flags.DEFINE_string("segment_dir", "data/segment/",
                       "Directory containing human segmentations.")
tf.flags.DEFINE_string("image_dir", "data/women_top/",
                       "Directory containing product and person images.")
tf.flags.DEFINE_string("test_label",
                       "data/viton_test_pairs.txt",
                       "File containing labels for testing.")
tf.flags.DEFINE_string("result_dir", "results/",
                       "Folder containing the results of testing.")

tf.flags.DEFINE_integer("begin", "0", "")
tf.flags.DEFINE_integer("end", "2032", "")
tf.logging.set_verbosity(tf.logging.INFO)


# preprocess images for testing
def _process_image(image_name, product_image_name, sess,
                   resize_width=192, resize_height=256):
  image_id = image_name[:-4]
  image = scipy.misc.imread(FLAGS.image_dir + image_name)
  prod_image = scipy.misc.imread(FLAGS.image_dir + product_image_name)
  segment_raw = sio.loadmat(os.path.join(
      FLAGS.segment_dir, image_id))["segment"]
  segment_raw = process_segment_map(segment_raw, image.shape[0], image.shape[1])
  pose_raw = sio.loadmat(os.path.join(FLAGS.pose_dir, image_id))
  pose_raw = extract_pose_keypoints(pose_raw)
  pose_raw = extract_pose_map(pose_raw, image.shape[0], image.shape[1])
  pose_raw = np.asarray(pose_raw, np.float32)

  body_segment, prod_segment, skin_segment = extract_segmentation(segment_raw)

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  prod_image = tf.image.convert_image_dtype(prod_image, dtype=tf.float32)

  image = tf.image.resize_images(image,
                                 size=[resize_height, resize_width],
                                 method=tf.image.ResizeMethod.BILINEAR)
  prod_image = tf.image.resize_images(prod_image,
                                      size=[resize_height, resize_width],
                                      method=tf.image.ResizeMethod.BILINEAR)

  body_segment = tf.image.resize_images(body_segment,
                                        size=[resize_height, resize_width],
                                        method=tf.image.ResizeMethod.BILINEAR,
                                        align_corners=False)
  skin_segment = tf.image.resize_images(skin_segment,
                                        size=[resize_height, resize_width],
                                        method=tf.image.ResizeMethod.BILINEAR,
                                        align_corners=False)

  prod_segment = tf.image.resize_images(prod_segment,
                                        size=[resize_height, resize_width],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  image = (image - 0.5) * 2.0
  prod_image = (prod_image - 0.5) * 2.0

  # using skin rbg
  skin_segment = skin_segment * image

  [image, prod_image, body_segment, prod_segment, skin_segment] = sess.run(
      [image, prod_image, body_segment, prod_segment, skin_segment])

  return image, prod_image, pose_raw, body_segment, prod_segment, skin_segment


def main(unused_argv):
  try:
    os.mkdir(FLAGS.result_dir)
  except:
    pass
  try:
    os.mkdir(FLAGS.result_dir + "/images/")
  except:
    pass
  try:
    os.mkdir(FLAGS.result_dir + "/tps/")
  except:
    pass

  # batch inference, can also be done one image per time.
  batch_size = 1
  image_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 3])
  prod_image_holder = tf.placeholder(
      tf.float32, shape=[batch_size, 256, 192, 3])
  body_segment_holder = tf.placeholder(
      tf.float32, shape=[batch_size, 256, 192, 1])
  prod_segment_holder = tf.placeholder(
      tf.float32, shape=[batch_size, 256, 192, 1])
  skin_segment_holder = tf.placeholder(
      tf.float32, shape=[batch_size, 256, 192, 3])
  pose_map_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 18])

  model = create_model(prod_image_holder, body_segment_holder,
                       skin_segment_holder, pose_map_holder,
                       prod_segment_holder, image_holder)

  images = np.zeros((batch_size, 256, 192, 3))
  prod_images = np.zeros((batch_size, 256, 192, 3))
  body_segments = np.zeros((batch_size, 256, 192, 1))
  prod_segments = np.zeros((batch_size, 256, 192, 1))
  skin_segments = np.zeros((batch_size, 256, 192, 3))
  pose_raws = np.zeros((batch_size, 256, 192, 18))

  saver = tf.train.Saver()
  with tf.Session() as sess:
    print("loading model from checkpoint")
    checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
    if checkpoint == None:
      checkpoint = FLAGS.checkpoint
    print(checkpoint)

    saver.restore(sess, checkpoint)

    # reading input data
    test_info = open(FLAGS.test_label).read().splitlines()
    for i in range(FLAGS.begin, FLAGS.end, batch_size):
      # loading batch data
      image_names = []
      product_image_names = []

      for j in range(i, i + batch_size):
        info = test_info[j].split()
        print(info)
        image_name = info[0]
        product_image_name = info[1]
        image_names.append(image_name)
        product_image_names.append(product_image_name)
        (image, prod_image, pose_raw,
         body_segment, prod_segment,
         skin_segment) = _process_image(image_name,
                                        product_image_name, sess)
        images[j-i] = image
        prod_images[j-i] = prod_image
        body_segments[j-i] = body_segment
        prod_segments[j-i] = prod_segment
        skin_segments[j-i] = skin_segment
        pose_raws[j-i] = pose_raw

      # inference
      feed_dict = {
          image_holder: images,
          prod_image_holder: prod_images,
          body_segment_holder: body_segments,
          skin_segment_holder: skin_segments,
          prod_segment_holder: prod_segments,
          pose_map_holder: pose_raws,
      }

      [image_output, mask_output, loss, step] = sess.run(
          [model.image_outputs,
           model.mask_outputs,
           model.gen_loss_content_L1,
           model.global_step],
          feed_dict=feed_dict)

      # write results
      for j in range(batch_size):
        scipy.misc.imsave(FLAGS.result_dir + ("images/%08d_" % step) +
                          image_names[j] + "_" + product_image_names[j] + '.png',
                          (image_output[j] / 2.0 + 0.5))
        scipy.misc.imsave(FLAGS.result_dir + ("images/%08d_" % step) +
                          image_names[j] + "_" + product_image_names[j] + '_mask.png',
                          np.squeeze(mask_output[j]))
        scipy.misc.imsave(FLAGS.result_dir + "images/" +
                          image_names[j], (images[j] / 2.0 + 0.5))
        scipy.misc.imsave(FLAGS.result_dir + "images/" +
                          product_image_names[j], (prod_images[j] / 2.0 + 0.5))
        sio.savemat(FLAGS.result_dir + "/tps/" + ("%08d_" % step) +
                    image_names[j] + "_" + product_image_names[j] + "_mask.mat",
                    {"mask": np.squeeze(mask_output[j])})

      # write html
      index_path = os.path.join(FLAGS.result_dir, "index.html")
      if os.path.exists(index_path):
        index = open(index_path, "a")
      else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th>"
                    "<th>output</th><th>target</th></tr>")
      for j in range(batch_size):
        index.write("<tr>")
        index.write("<td>%d %d</td>" % (step, i + j))
        index.write("<td>%s %s</td>" % (image_names[j], product_image_names[j]))
        index.write("<td><img src='images/%s'></td>" % image_names[j])
        index.write("<td><img src='images/%s'></td>" % product_image_names[j])
        index.write("<td><img src='images/%08d_%s'></td>" %
                    (step, image_names[j] + "_" + product_image_names[j] + '.png'))
        index.write("<td><img src='images/%08d_%s'></td>" %
                    (step, image_names[j] + "_" + product_image_names[j] + '_mask.png'))
        index.write("</tr>")

if __name__ == "__main__":
  tf.app.run()
