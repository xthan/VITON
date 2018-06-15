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

""" Util functions of virtual try-on model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy
import numpy as np
import tensorflow as tf

from scipy.misc import imresize


def extract_pose_keypoints(pose):
  pose_keypoints = - np.ones((18, 2), dtype=int)
  for i in range(18):
    if pose['subset'][0, i] != -1:
      pose_keypoints[i, :] = pose['candidate'][int(pose['subset'][0, i]), :2]
  return pose_keypoints  # only return the coordinates


def extract_pose_map(pose_keypoints, h, w, resize_h=256.0, resize_w=192.0):
  """Given 18 * 2 keypoints, and imge size, return a resize_h*resize_w*18 map"""
  pose_keypoints = np.asarray(pose_keypoints, np.float32)
  pose_keypoints[:, 0] = pose_keypoints[:, 0] * resize_w / float(w)
  pose_keypoints[:, 1] = pose_keypoints[:, 1] * resize_h / float(h)
  pose_keypoints = np.asarray(pose_keypoints, np.int)
  pose_map = np.zeros((int(resize_h), int(resize_w), 18), np.bool)
  for i in range(18):
    if pose_keypoints[i, 0] < 0:
      continue
    t = np.max((pose_keypoints[i, 1] - 5, 0))
    b = np.min((pose_keypoints[i, 1] + 5, h - 1))
    l = np.max((pose_keypoints[i, 0] - 5, 0))
    r = np.min((pose_keypoints[i, 0] + 5, w - 1))
    pose_map[t:b+1, l:r+1, i] = True
  return pose_map


def extract_segmentation(segment):
  """Given semantic segmentation map, extract the body part."""
  product_segmentation = tf.cast(tf.equal(segment, 5), tf.float32)


  skin_segmentation = (tf.cast(tf.equal(segment, 1), tf.float32) +
                       tf.cast(tf.equal(segment, 2), tf.float32) +
                       tf.cast(tf.equal(segment, 4), tf.float32) +
                       tf.cast(tf.equal(segment, 13), tf.float32))

  body_segmentation = (1.0 - tf.cast(tf.equal(segment, 0), tf.float32) -
                          skin_segmentation)


  # Extend the axis
  product_segmentation = tf.expand_dims(product_segmentation, -1)
  body_segmentation = tf.expand_dims(body_segmentation, -1)
  skin_segmentation = tf.expand_dims(skin_segmentation, -1)

  body_segmentation = tf.image.resize_images(body_segmentation,
                                size=[16, 12],
                                method=tf.image.ResizeMethod.AREA,
                                align_corners=False)

  return body_segmentation, product_segmentation, skin_segmentation




def process_segment_map(segment, h, w):
  """Extract segment maps."""
  segment = np.asarray(segment, dtype=np.uint8)
  segment = imresize(segment, (h, w), interp='nearest')
  return segment


def extract_pose_representation(pose_keypoints, h, w, resize_h, resize_w):
  """Given pose keypoints, return a h*w*18 [0,1] map to represent the pose."""
  resize_h_ratio = float(resize_h) / tf.cast(h, tf.float32)
  resize_w_ratio = float(resize_w) / tf.cast(w, tf.float32)
  invisible_points = tf.less(pose_keypoints[:, 0], 0.0)
  pose_keypoints = pose_keypoints * tf.Variable((resize_w_ratio,
                                                 resize_h_ratio),
                                                tf.float32)

  pose_keypoints = tf.cast(pose_keypoints, tf.int64)
  # pose_representation = tf.zeros((h, w, 18))

  pose_representation = tf.one_hot(pose_keypoints, 3)
  return pose_representation


def parse_tf_example(serialized, stage=""):
  """Parses a tensorflow.SequenceExample into an image and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized TF Example.
    stage; If "tps_points", return a different set of variables.
  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    encoded_prod_image: A JPEG encoded image string of the product image.
    body_segment: A h X w [0,1] Tensor indicating the body part.
    product_segment: A h X w [0,1] Tensor indicating the clothing part.
    skin_segment: A h X w Tensor indicating the skin part.
    pose_map: A 256 X 256 * 18 Tensor indicating pose.
  """
  features = tf.parse_single_example(
      serialized,
      features={
          "image_id": tf.FixedLenFeature([], tf.string),
          "height": tf.FixedLenFeature([], tf.int64),
          "width": tf.FixedLenFeature([], tf.int64),
          "image": tf.FixedLenFeature([], tf.string),
          "product_image": tf.FixedLenFeature([], tf.string),
          "pose_map": tf.FixedLenFeature([], tf.string),
          "segment_map": tf.FixedLenFeature([], tf.string),
          "tps_control_points": tf.VarLenFeature(tf.float32),
          "num_keypoints": tf.FixedLenFeature([], tf.int64),
          "keypoints": tf.VarLenFeature(tf.float32),
          "prod_keypoints": tf.VarLenFeature(tf.float32),
      }
  )
  encoded_product_image = features["product_image"]
  encoded_image = features["image"]

  height = tf.cast(features["height"], tf.int32)
  width = tf.cast(features["width"], tf.int32)
  pose_map = tf.decode_raw(features["pose_map"], tf.uint8)
  pose_map = tf.cast(pose_map, tf.float32)
  pose_map = tf.reshape(pose_map, tf.stack([256, 192, 18]))
  segment_map = tf.decode_raw(features["segment_map"], tf.uint8)
  segment_map = tf.reshape(segment_map, tf.stack([height, width]))
  body_segment, prod_segment, skin_segment = extract_segmentation(segment_map)

  if stage != "tps_points":
    return (encoded_image, encoded_product_image, body_segment, prod_segment,
          skin_segment, pose_map, features["image_id"])

  # TPS control points reshape
  tps_points = features["tps_control_points"]
  tps_points = tf.sparse_tensor_to_dense(tps_points, default_value=0.)
  tps_points = tf.reshape(tps_points, tf.stack([2,10,10]))
  tps_points = tf.transpose(tf.reshape(tps_points, tf.stack([2, 100]))) * 2 - 1
  return (encoded_image, encoded_product_image, body_segment, prod_segment,
          skin_segment, pose_map, features["image_id"], tps_points)

def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_memory factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  tf.summary.scalar(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue


def distort_image(image, thread_id):
  """Perform random distortions on an image.

  Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.

  Returns:
    distorted_image: A float32 Tensor of shape [height, width, 3] with values in
      [0, 1].
  """
  # Randomly flip horizontally.
  with tf.name_scope("flip_horizontal", values=[image]):
    image = tf.image.random_flip_left_right(image)

  return image



def process_image(encoded_image,
                  encoded_prod_image,
                  body_segment,
                  prod_segment,
                  skin_segment,
                  pose_map,
                  is_training,
                  height=256,
                  width=192,
                  resize_height=256,
                  resize_width=192,
                  thread_id=0,
                  image_format="jpeg",
                  zero_one_mask=True,
                  different_image_size=False):
  """Decode an image, resize and apply random distortions.

  In training, images are distorted slightly differently depending on thread_id.

  Args:
    encoded_image: String Tensor containing the image.
    encoded_product_image: String Tensor containing the product image.
    body_segment: Matrix containing the segmentation of body part.
    prod_segment: Matrix containing the segmentation of product part.
    skin_segment: Matrix containing the segmentation of product part.
    pose_map: Matrix containing the pose keypoints.
    is_training: Boolean; whether preprocessing for training or eval.
    height: Height of the output image.
    width: Width of the output image.
    resize_height: If > 0, resize height before crop to final dimensions.
    resize_width: If > 0, resize width before crop to final dimensions.
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
    image_format: "jpeg" or "png".
    zero_one_mask: True if use 0,1 mask, False if use -1,1 mask.
    different_image_size: True to output image of size heigt, width
  Returns:
    A float32 Tensor of shape [height, width, 3] with values in [-1, 1].
    pose_map: A Tensor representing the pose.

  Raises:
    ValueError: If image_format is invalid.
  """
  # Helper function to log an image summary to the visualizer. Summaries are
  # only logged in thread 0.
  def image_summary(name, image):
    return
    # Do not do summary inside .
    # if not thread_id:
    #   tf.summary.image(name, tf.expand_dims(image, 0))

  # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
  with tf.name_scope("decode", values=[encoded_image]):
    if image_format == "jpeg":
      image = tf.image.decode_jpeg(encoded_image, channels=3)
      prod_image = tf.image.decode_jpeg(encoded_prod_image, channels=3)
    elif image_format == "png":
      image = tf.image.decode_png(encoded_image, channels=3)
      prod_image = tf.image.decode_png(encoded_prod_image, channels=3)
    else:
      raise ValueError("Invalid image format: %s" % image_format)

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  prod_image = tf.image.convert_image_dtype(prod_image, dtype=tf.float32)
  image_summary("original_image", image)
  image_summary("original_prod_image", prod_image)
  image_summary("original_body_seg", body_segment)
  image_summary("original_prod_seg", prod_segment)
  image_summary("original_skin_seg", skin_segment)

  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  if different_image_size:
    image = tf.image.resize_images(image,
                                   size=[height, width],
                                   method=tf.image.ResizeMethod.BILINEAR)
    prod_image = tf.image.resize_images(prod_image,
                                        size=[height, width],
                                        method=tf.image.ResizeMethod.BILINEAR)
  else:
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
  
  image_summary("final_image", image)
  image_summary("final_prod_image", prod_image)
  image_summary("final_body_seg", body_segment)
  image_summary("final_prod_seg", prod_segment)
  image_summary("final_skin_seg", skin_segment)

  # Rescale to [-1,1] instead of [0, 1]
  image = (image - 0.5) * 2.0
  prod_image = (prod_image - 0.5) * 2.0
  
  # instead of using 16x12 skin segment, now using skin rbg
  skin_segment = skin_segment * image
    
  if not zero_one_mask:
    body_segment = (body_segment - 0.5) * 2.0
    prod_segment = (prod_segment - 0.5) * 2.0
    skin_segment = (skin_segment - 0.5) * 2.0
    pose_map = (pose_map - 0.5) * 2.0
  
  return image, prod_image, body_segment, prod_segment, skin_segment, pose_map



def conv(batch_input, out_channels, stride):
  with tf.variable_scope("conv"):
    in_channels = batch_input.get_shape()[3]
    filter = tf.get_variable("filter",
                             [4, 4, in_channels, out_channels],
                             dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.02))
    # [batch, in_height, in_width, in_channels]
    # [filter_width, filter_height, in_channels, out_channels]
    #   => [batch, out_height, out_width, out_channels]
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [
                          1, 1], [0, 0]], mode="CONSTANT")
    conv = tf.nn.conv2d(padded_input, filter, [
                        1, stride, stride, 1], padding="VALID")
    return conv


def final_conv(batch_input, out_channels=1, stride=1):
  with tf.variable_scope("conv"):
    in_channels = batch_input.get_shape()[3]
    filter = tf.get_variable("filter",
                         [4, 3, in_channels, out_channels],
                         dtype=tf.float32,
                         initializer=tf.random_normal_initializer(0, 0.02))

    conv = tf.nn.conv2d(batch_input, filter, [
                        1, stride, stride, 1], padding="VALID")
    return conv


def lrelu(x, a=0.2):
  with tf.name_scope("lrelu"):
    # adding these together creates the leak part and linear part
    # then cancels them out by subtracting/adding an absolute value term
    # leak: a*x/2 - a*abs(x)/2
    # linear: x/2 + abs(x)/2

    # this block looks like it has 2 inputs on the graph unless we do this
    x = tf.identity(x)
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


# seperate batch norm training and testing
def batch_norm(inputs, is_training=True, decay=0.999):
  with tf.variable_scope("batchnorm"):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    epsilon = 1e-5
    if is_training:
      batch_mean, batch_var = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=False)
      train_mean = tf.assign(pop_mean,
                             pop_mean * decay + batch_mean * (1 - decay))
      train_var = tf.assign(pop_var,
                            pop_var * decay + batch_var * (1 - decay))
      with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(inputs,
                                         batch_mean, batch_var, beta, scale, epsilon)
    else:
      return tf.nn.batch_normalization(inputs,
                                       pop_mean, pop_var, beta, scale, epsilon)

def deconv(batch_input, out_channels):
  with tf.variable_scope("deconv"):
    batch, in_height, in_width, in_channels = [
        int(d) for d in batch_input.get_shape()]
    filter = tf.get_variable("filter",
                             [4, 4, out_channels, in_channels],
                             dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0, 0.02))
    # [batch, in_height, in_width, in_channels]
    #  [filter_width, filter_height, out_channels, in_channels]
    #   => [batch, out_height, out_width, out_channels]
    conv = tf.nn.conv2d_transpose(batch_input,
                          filter,
                          [batch, in_height * 2, in_width * 2, out_channels],
                          [1, 2, 2, 1],
                          padding="SAME")
    return conv


# some functions used in stage2
def build_net(ntype, nin, nwb=None, name=None):
  if ntype == 'conv':
    return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1],
                                   padding='SAME', name=name)+nwb[1])
  elif ntype == 'pool':
    return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def get_weight_bias(vgg_layers, i):
  weights = vgg_layers[i][0][0][2][0][0]
  weights = tf.constant(weights)
  bias = vgg_layers[i][0][0][2][0][1]
  bias = tf.constant(np.reshape(bias, (bias.size)))
  return weights, bias


def build_vgg19(input, model_path, reuse=False):
  if reuse:
    tf.get_variable_scope().reuse_variables()
  net = {}
  vgg_rawnet = scipy.io.loadmat(model_path)
  vgg_layers = vgg_rawnet['layers'][0]
  # input is [-1,1], we need to scale it to [0,255] and minize the VGG means.
  input = (input + 1.0) / 2 * 255.0
  mean = np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
  net['input'] = input - mean
  net['conv1_1'] = build_net('conv', net['input'],
                           get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
  net['conv1_2'] = build_net('conv', net['conv1_1'],
                           get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
  net['pool1'] = build_net('pool', net['conv1_2'])
  net['conv2_1'] = build_net(
      'conv', net['pool1'], get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
  net['conv2_2'] = build_net('conv', net['conv2_1'],
                          get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
  net['pool2'] = build_net('pool', net['conv2_2'])
  net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(
      vgg_layers, 10), name='vgg_conv3_1')
  net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(
      vgg_layers, 12), name='vgg_conv3_2')
  net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(
      vgg_layers, 14), name='vgg_conv3_3')
  net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(
      vgg_layers, 16), name='vgg_conv3_4')
  net['pool3'] = build_net('pool', net['conv3_4'])
  net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(
      vgg_layers, 19), name='vgg_conv4_1')
  net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(
      vgg_layers, 21), name='vgg_conv4_2')
  net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(
      vgg_layers, 23), name='vgg_conv4_3')
  net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(
      vgg_layers, 25), name='vgg_conv4_4')
  net['pool4'] = build_net('pool', net['conv4_4'])
  net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(
      vgg_layers, 28), name='vgg_conv5_1')
  net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(
      vgg_layers, 30), name='vgg_conv5_2')
  net['conv5_3'] = build_net('conv', net['conv5_2'], get_weight_bias(
      vgg_layers, 32), name='vgg_conv5_3')
  net['conv5_4'] = build_net('conv', net['conv5_3'], get_weight_bias(
      vgg_layers, 34), name='vgg_conv5_4')
  net['pool5'] = build_net('pool', net['conv5_4'])
  return net

# image and web summaries.
def save_images(fetches, image_dict, output_dir, step=None):
  # ["image", "product_image", "body_segment",
  # "prod_segment", "skin_segment", "outputs"]
  image_dir = os.path.join(output_dir, "images")
  if not os.path.exists(image_dir):
    os.makedirs(image_dir)

  filesets = []
  for i, in_path in enumerate(fetches["paths"]):
    if i >= 1:
      # continue
      break # only show up to 1 images for batch
    name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
    fileset = {"name": name, "step": step}
    for kind in image_dict:
      filename = name + "-" + kind + ".png"
      if step is not None:
        filename = "%08d-%s" % (step, filename)
      fileset[kind] = filename
      out_path = os.path.join(image_dir, filename)
      contents = fetches[kind][i]
      with open(out_path, "wb") as f:
        f.write(contents)
    filesets.append(fileset)
  return filesets


def append_index(filesets, image_dict, output_dir, step=False):
  # ["image", "product_image", "body_segment",
  # "prod_segment", "skin_segment", "outputs"]
  index_path = os.path.join(output_dir, "index.html")
  if os.path.exists(index_path):
    index = open(index_path, "a")
  else:
    index = open(index_path, "w")
    index.write("<html><body><table><tr>")
    if step:
      index.write("<th>step</th>")
    index.write("<th>name</th><th>input</th>"
      "<th>output</th><th>target</th></tr>")

  for fileset in filesets:
    index.write("<tr>")

    if step:
      index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    for kind in image_dict:
      index.write("<td><img src='images/%s'></td>" % fileset[kind])

    index.write("</tr>")
  return index_path



def compute_error(real, fake, mask=None):
  if mask == None:
    return tf.reduce_mean(tf.abs(fake - real))  # simple loss
  else:
    _, h, w, _ = real.get_shape().as_list()
    sampled_mask = tf.image.resize_images(mask, (h, w),
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.reduce_mean(tf.abs(fake - real) * sampled_mask)  # simple loss


