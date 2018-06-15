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

""" Test for Stage 2: from product image + warpped image => refined image.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import *

import collections
from model_zalando_tps_warp import create_refine_generator
import os
import time
from tps_transformer import tps_stn

import numpy as np
import scipy.io as sio
import scipy.misc
import tensorflow as tf
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("image_dir", "data/women_top/",
                       "Directory containing product and person images.")
tf.flags.DEFINE_string("test_label",
                       "data/viton_test_pairs.txt",
                       "File containing labels for testing.")
tf.flags.DEFINE_string("result_dir", "results/stage2/",
                       "Folder containing the results of testing.")
tf.flags.DEFINE_string("coarse_result_dir", "results/stage1",
                  "Folder containing the results of stage1 (coarse) results.")

tf.flags.DEFINE_integer("begin", "0", "")
tf.flags.DEFINE_integer("end", "2032", "")


tf.logging.set_verbosity(tf.logging.INFO)



def deprocess_image(image, mask01=False):
  if not mask01:
    image = image / 2 + 0.5
  return image

def process_one_image(image, resize_height, resize_width, if_zero_one=False):
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if if_zero_one:
    return image
  image = tf.image.resize_images(image,
                                 size=[resize_height, resize_width],
                                 method=tf.image.ResizeMethod.BILINEAR)
  return (image - 0.5) * 2.0

# preprocess images for testing
def _process_image(image_name, product_image_name, sess,
                   resize_width=192, resize_height=256):
  image_id = image_name[:-4]
  image = scipy.misc.imread(FLAGS.image_dir + image_name)
  prod_image = scipy.misc.imread(FLAGS.image_dir + product_image_name)
  # sorry for the hard coded file path.
  coarse_image = scipy.misc.imread(FLAGS.coarse_result_dir +
                                   "/images/00015000_" +
                                   image_name + "_" +
                                   product_image_name + ".png")
  mask_output = scipy.misc.imread(FLAGS.coarse_result_dir +
                                  "/images/00015000_" +
                                  image_name + "_" +
                                  product_image_name + "_mask.png")
  image = process_one_image(image, resize_height, resize_width)
  prod_image = process_one_image(prod_image, resize_height, resize_width)
  coarse_image = process_one_image(coarse_image, resize_height, resize_width)
  mask_output = process_one_image(mask_output, resize_height,
                                  resize_width, True)
  # TPS transform
  # Here we use control points to generate 
  # We tried to learn the control points, but the network refuses to converge.
  tps_control_points = sio.loadmat(FLAGS.coarse_result_dir +
                                   "/tps/00015000_" +
                                   image_name + "_" +
                                   product_image_name +
                                   "_tps.mat")
  v = tps_control_points["control_points"]
  nx = v.shape[1]
  ny = v.shape[2]
  v = np.reshape(v, -1)
  v = np.transpose(v.reshape([1,2,nx*ny]), [0,2,1]) * 2 -1
  p = tf.convert_to_tensor(v, dtype=tf.float32)
  img = tf.reshape(prod_image, [1,256,192,3])

  tps_image = tps_stn(img, nx, ny, p, [256,192,3])

  tps_mask = tf.cast(tf.less(tf.reduce_sum(tps_image, -1), 3*0.95), tf.float32)

  [image, prod_image, coarse_image, tps_image, mask_output, tps_mask] = sess.run(
              [image, prod_image, coarse_image, tps_image, mask_output, tps_mask])

  return image, prod_image, coarse_image, tps_image, mask_output, tps_mask

def main(unused_argv):
  try:
    os.mkdir(FLAGS.result_dir)
  except:
    pass
  try:
      os.mkdir(FLAGS.result_dir + "/images/")
  except:
    pass

  batch_size = 1

  # Feed into the refine module
  image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
  prod_image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
  prod_mask_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,1])
  coarse_image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
  tps_image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])  

  with tf.variable_scope("refine_generator") as scope:
    select_mask = create_refine_generator(tps_image_holder,
                                          coarse_image_holder)
    select_mask = select_mask * prod_mask_holder
    model_image_outputs = (select_mask * tps_image_holder +
                           (1 - select_mask) * coarse_image_holder)

  saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() 
                                   if var.name.startswith("refine_generator")])
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
      print(i)
      images = np.zeros((batch_size,256,192,3))
      prod_images = np.zeros((batch_size,256,192,3))
      coarse_images = np.zeros((batch_size,256,192,3))
      tps_images = np.zeros((batch_size,256,192,3))
      mask_outputs = np.zeros((batch_size,256,192,1))

      image_names = []
      product_image_names = []

      for j in range(i, i + batch_size):
        info = test_info[j].split()
        print(info)
        image_name = info[0]
        product_image_name = info[1]
        image_names.append(image_name)
        product_image_names.append(product_image_name)
        try:
          (image, prod_image, coarse_image,
           tps_image, mask_output, tps_mask) = _process_image(image_name,
                                                   product_image_name, sess)
        except:
          continue

        images[j-i] = image
        prod_images[j-i] = prod_image
        coarse_images[j-i] = coarse_image
        tps_images[j-i] = tps_image
        mask_outputs[j-i] = np.expand_dims(mask_output, -1)

      # inference
      feed_dict = {
        image_holder: images,
        prod_image_holder: prod_images,
        coarse_image_holder: coarse_images,
        tps_image_holder: tps_images,
        prod_mask_holder: mask_outputs,
      }

      [image_output, sel_mask] = sess.run([model_image_outputs, select_mask],
                                feed_dict=feed_dict)

      # write results
      for j in range(batch_size):
        step = 0
        scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j] +
                          "_" + product_image_names[j] + '_tps.png',
                          (tps_images[j] / 2.0 + 0.5))
        scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j] +
                          "_" + product_image_names[j] + '_coarse.png',
                          (coarse_images[j] / 2.0 + 0.5))
        scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j] +
                          "_" + product_image_names[j] + '_mask.png',
                          np.squeeze(mask_outputs[j]))
        scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j] +
                          "_" + product_image_names[j] + '_final.png',
                          (image_output[j]) / 2.0 + 0.5)
        scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j] +
                          "_" + product_image_names[j] + '_sel_mask.png',
                          np.squeeze(sel_mask[j]))
        scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j],
                          (images[j] / 2.0 + 0.5))
        scipy.misc.imsave(FLAGS.result_dir + "images/"+ product_image_names[j],
                          (prod_images[j] / 2.0 + 0.5))

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
        index.write("<td>%s %s</td>" % (image_names[j],
                                          product_image_names[j]))
        index.write("<td><img src='images/%s'></td>" % image_names[j])
        index.write("<td><img src='images/%s'></td>" % product_image_names[j])
        index.write("<td><img src='images/%s'></td>" % 
           (image_names[j] + "_" + product_image_names[j] + '_tps.png'))
        index.write("<td><img src='images/%s'></td>" % 
          (image_names[j] + "_" + product_image_names[j] + '_coarse.png'))
        index.write("<td><img src='images/%s'></td>" % 
           (image_names[j] + "_" + product_image_names[j] + '_mask.png'))
        index.write("<td><img src='images/%s'></td>" % 
           (image_names[j] + "_" + product_image_names[j] + '_final.png'))
        index.write("<td><img src='images/%s'></td>" % 
           (image_names[j] + "_" + product_image_names[j] + '_sel_mask.png'))
        index.write("</tr>")

if __name__ == "__main__":
  tf.app.run()
