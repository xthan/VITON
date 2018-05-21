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
    pose predict product segment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import *

import collections
from baseline_cagan import create_model
import os
import time

import numpy as np
import scipy.io as sio
import scipy.misc
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("pose_dir", "../../../dataset/zalando/pose/women_top/",
                       "Directory containing poses.")
tf.flags.DEFINE_string("segment_dir", "../../../dataset/zalando/segment/women_top/",
                       "Directory containing segmentations.")
tf.flags.DEFINE_string("image_dir", "../../../dataset/zalando/women_top/",
                       "Directory containing segmentations.")
tf.flags.DEFINE_string("test_label",
                       "../../prepare_data/label/zalando_women_test_pairs_new.txt",
                       "File containing labels for testing.")
tf.flags.DEFINE_string("result_dir", "results/",
                       "Folder containing the results of testing.")

tf.flags.DEFINE_integer("begin", "0", "")
tf.flags.DEFINE_integer("end", "2032", "")


tf.logging.set_verbosity(tf.logging.INFO)



def deprocess_image(image, mask01=False):
  if not mask01:
    image = image / 2 + 0.5
  # return tf.map_fn(tf.image.encode_png, tf.image.convert_image_dtype(image, dtype=tf.uint8),dtype=tf.string)
  # return tf.image.convert_image_dtype(image, dtype=tf.uint8)
  return image




# preprocess images for testing
def _process_image(image_name, product_image_name1, product_image_name2, sess,
                   resize_width=192, resize_height=256):
  image_id = image_name[:-4]
  image = scipy.misc.imread(FLAGS.image_dir + image_name)
  prod_image1 = scipy.misc.imread(FLAGS.image_dir + product_image_name1)
  prod_image2 = scipy.misc.imread(FLAGS.image_dir + product_image_name2)
  
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  prod_image1 = tf.image.convert_image_dtype(prod_image1, dtype=tf.float32)
  prod_image2 = tf.image.convert_image_dtype(prod_image2, dtype=tf.float32)

  image = tf.image.resize_images(image,
                                 size=[resize_height, resize_width],
                                 method=tf.image.ResizeMethod.BILINEAR)
  prod_image1 = tf.image.resize_images(prod_image1,
                                      size=[resize_height, resize_width],
                                      method=tf.image.ResizeMethod.BILINEAR)
  prod_image2 = tf.image.resize_images(prod_image2,
                                      size=[resize_height, resize_width],
                                      method=tf.image.ResizeMethod.BILINEAR)

  image = (image - 0.5) * 2.0
  prod_image1 = (prod_image1 - 0.5) * 2.0
  prod_image2 = (prod_image2 - 0.5) * 2.0

  [image, prod_image1, prod_image2] = sess.run([image, prod_image1, prod_image2])
  return image, prod_image1, prod_image2

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
      os.mkdir(FLAGS.result_dir + "/tmp/")
  except:
    pass

  batch_size = 16
  image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
  prod_image_holder1 = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
  prod_image_holder2 = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])

  model = create_model(image_holder, prod_image_holder1, prod_image_holder2)

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
    # for info in test_info[10:11]:
    for i in range(FLAGS.begin,FLAGS.end,batch_size): # range(0, len(test_info) - 16, 16):
      # loading batch data
      print(i)
      images = np.zeros((batch_size,256,192,3))
      prod_images1 = np.zeros((batch_size,256,192,3))
      prod_images2 = np.zeros((batch_size,256,192,3))
      image_names = []
      product_image_names1 = []
      product_image_names2 = []

      for j in range(i, i + batch_size):
        info = test_info[j].split()
        print(info)
        image_name = info[0]
        ########
        product_image_name1 = info[0][:-5] + "1.jpg"
        product_image_name2 = info[1]
        image_names.append(image_name)
        product_image_names1.append(product_image_name1)
        product_image_names2.append(product_image_name2)

        (image, prod_image1, prod_image2) = _process_image(image_name,
                                              product_image_name1,
                                              product_image_name2,
                                              sess)
        images[j-i] = image
        prod_images1[j-i] = prod_image1
        prod_images2[j-i] = prod_image2

      # inference
      feed_dict = {
        image_holder: images,
        prod_image_holder1: prod_images1,
        prod_image_holder2: prod_images2,
      }

      [image_output, step] = sess.run(
                              [model.image_outputs,
                               model.global_step],
                               feed_dict=feed_dict)


      # write results
      for j in range(batch_size):
        scipy.misc.imsave(FLAGS.result_dir + ("images/%08d_" % step) + image_names[j] + "_" + product_image_names2[j] + '.png', (image_output[j] / 2.0 + 0.5))
        scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j], (images[j] / 2.0 + 0.5))
        scipy.misc.imsave(FLAGS.result_dir + "images/"+ product_image_names1[j], (prod_images1[j] / 2.0 + 0.5))
        scipy.misc.imsave(FLAGS.result_dir + "images/"+ product_image_names2[j], (prod_images2[j] / 2.0 + 0.5))

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
        index.write("<td>%s %s</td>" % (image_names[j], product_image_names2[j]))
        index.write("<td><img src='images/%s'></td>" % image_names[j])
        index.write("<td><img src='images/%s'></td>" % product_image_names1[j])
        index.write("<td><img src='images/%s'></td>" % product_image_names2[j])
        index.write("<td><img src='images/%08d_%s'></td>" % (step, image_names[j] + "_" + product_image_names2[j] + '.png'))
        index.write("</tr>")

if __name__ == "__main__":
  tf.app.run()
