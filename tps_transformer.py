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


""" Thin Plate Spline Transformer Network.
Code adopted from https://github.com/iwyoo/TPS_STN-tensorflow/blob/master/TPS_STN.py
"""

from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
# np.set_printoptions(threshold=np.nan)
import scipy.io as sio


def tps_stn(U, nx, ny, cp, out_size, is_points=False, points=None):
  """Thin Plate Spline Spatial Transformer Layer
  TPS control points are arranged in a regular grid.

  U : float Tensor
      shape [num_batch, height, width, num_channels].
  nx : int
      The number of control points on x-axis
  ny : int
      The number of control points on y-axis
  cp : float Tensor
      control points. shape [num_batch, nx*ny, 2].
  out_size: tuple of two ints
      The size of the output of the network (height, width)
  is_points: true if transform points instead of image.
  points: keypoints coordinates.
  ----------
  Reference :
    https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
  """

  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
    rep = tf.cast(rep, 'int32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  def _interpolate(im, x, y, out_size):
    # constants
    num_batch = tf.shape(im)[0]
    height = tf.shape(im)[1]
    width = tf.shape(im)[2]
    channels = tf.shape(im)[3]

    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    out_height = out_size[0]
    out_width = out_size[1]
    # clip coordinates to [-1, 1]
    x = tf.clip_by_value(x, -1, 1)
    y = tf.clip_by_value(y, -1, 1)
    # scale coordinates from [-1, 1] to [0, width/height-1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    # do sampling
    x0_f = tf.floor(x)
    y0_f = tf.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    x0 = tf.cast(x0_f, 'int32')
    y0 = tf.cast(y0_f, 'int32')
    x1 = tf.cast(tf.minimum(x1_f, width_f - 1), 'int32')
    y1 = tf.cast(tf.minimum(y1_f, height_f - 1), 'int32')

    dim2 = width
    dim1 = width*height
    base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, tf.stack([-1, channels]))
    Ia = tf.gather(im_flat, idx_a)
    Ib = tf.gather(im_flat, idx_b)
    Ic = tf.gather(im_flat, idx_c)
    Id = tf.gather(im_flat, idx_d)

    # and finally calculate interpolated values
    wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
    wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
    wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
    wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
    output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return output

  def _meshgrid(height, width, fp):
    x_t = tf.matmul(
        tf.ones(shape=tf.stack([height, 1])),
        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(
        tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
        tf.ones(shape=tf.stack([1, width])))

    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))

    x_t_flat_b = tf.expand_dims(x_t_flat, 0)  # [1, 1, h*w]
    y_t_flat_b = tf.expand_dims(y_t_flat, 0)  # [1, 1, h*w]

    num_batch = tf.shape(fp)[0]
    px = tf.expand_dims(fp[:, :, 0], 2)  # [n, nx*ny, 1]
    py = tf.expand_dims(fp[:, :, 1], 2)  # [n, nx*ny, 1]
    d = tf.pow(x_t_flat_b - px, 2.) + tf.pow(y_t_flat_b - py, 2.)
    r = d * tf.log(d + 1e-12)  # [n, nx*ny, h*w]
    x_t_flat_g = tf.tile(x_t_flat_b, tf.stack([num_batch, 1, 1]))  # [n, 1, h*w]
    y_t_flat_g = tf.tile(y_t_flat_b, tf.stack([num_batch, 1, 1]))  # [n, 1, h*w]
    ones = tf.ones_like(x_t_flat_g)  # [n, 1, h*w]

    grid = tf.concat([ones, x_t_flat_g, y_t_flat_g, r], 1)  # [n, nx*ny+3, h*w]
    return grid


  def _transform(T, fp, input_dim, out_size):
    num_batch = input_dim.get_shape()[0]
    height = tf.shape(input_dim)[1]
    width = tf.shape(input_dim)[2]
    num_channels = input_dim.get_shape()[3]

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    out_height = out_size[0]
    out_width = out_size[1]
    grid = _meshgrid(out_height, out_width, fp)  # [2, h*w]

    # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
    T_g = tf.matmul(T, grid)  # MARK
    x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
    x_s_flat = tf.reshape(x_s, [-1])
    y_s_flat = tf.reshape(y_s, [-1])

    input_transformed = _interpolate(
        input_dim, x_s_flat, y_s_flat, out_size)

    output = tf.reshape(
        input_transformed,
        tf.stack([num_batch, out_height, out_width, num_channels]))
    return output


  def _point_transform(T, fp, points, out_size):
    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    height = out_size[0]
    width = out_size[1]

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    num_batch = tf.shape(fp)[0]
    x_t = points[:,::2] # [n, num_points]
    y_t = points[:,1::2] # [n, num_points]
    x_t_b = tf.expand_dims(x_t, 1)  # [n, 1, num_points]
    y_t_b = tf.expand_dims(y_t, 1)  # [n, 1, num_points]

    px = tf.expand_dims(fp[:, :, 0], 2)  # [n, nx*ny, 1]
    py = tf.expand_dims(fp[:, :, 1], 2)  # [n, nx*ny, 1]
    

    d = tf.pow(x_t_b - px, 2.) + tf.pow(y_t_b - py, 2.) # [n, nx*ny, num_points]
    r = d * tf.log(d + 1e-12)  # [n, nx*ny, num_points]
    ones = tf.ones_like(x_t_b)  # [n, 1, num_points]


    grid = tf.concat([ones, x_t_b, y_t_b, r], 1)  # [n, nx*ny+3, num_points]



    x_t = tf.matmul(
        tf.ones(shape=tf.stack([height, 1])),
        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(
        tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
        tf.ones(shape=tf.stack([1, width])))

    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))

    x_t_flat_b = tf.expand_dims(x_t_flat, 0)  # [1, 1, h*w]
    y_t_flat_b = tf.expand_dims(y_t_flat, 0)  # [1, 1, h*w]

    num_batch = tf.shape(fp)[0]
    px = tf.expand_dims(fp[:, :, 0], 2)  # [n, nx*ny, 1]
    py = tf.expand_dims(fp[:, :, 1], 2)  # [n, nx*ny, 1]
    d = tf.pow(x_t_flat_b - px, 2.) + tf.pow(y_t_flat_b - py, 2.)
    r = d * tf.log(d + 1e-12)  # [n, nx*ny, h*w]
    x_t_flat_g = tf.tile(x_t_flat_b, tf.stack([num_batch, 1, 1]))  # [n, 1, h*w]
    y_t_flat_g = tf.tile(y_t_flat_b, tf.stack([num_batch, 1, 1]))  # [n, 1, h*w]
    ones = tf.ones_like(x_t_flat_g)  # [n, 1, h*w]

    grid = tf.concat([ones, x_t_flat_g, y_t_flat_g, r], 1)  # [n, nx*ny+3, h*w]


    # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
    T_g = tf.matmul(T, grid)  # MARK
    x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
    T_g = tf.concat([x_s, y_s], 1)
    T_g = tf.transpose(T_g, [0, 2, 1])
    T_g = tf.reshape(T_g, [num_batch, -1])  # MARK

  def _solve_system(cp, nx, ny):
    gx = 2. / (nx - 1)  # grid x size
    gy = 2. / (ny - 1) # grid y size
    cx = -1. # x coordinate
    cy = -1. # y coordinate

    p_ = np.empty([nx*ny, 3], dtype='float32')
    i = 0
    for _ in range(ny):
      for _ in range(nx):
        p_[i, :] = 1, cx, cy
        i += 1
        cx += gx
      cx = -1.
      cy += gy
    p_1 = p_.reshape([nx*ny, 1, 3])
    p_2 = p_.reshape([1, nx*ny, 3])
    d = np.sqrt(np.sum((p_1-p_2)**2, 2))  # [nx*ny, nx*ny]
    r = d*d*np.log(d*d+1e-12)
    W = np.zeros([nx*ny+3, nx*ny+3], dtype='float32')
    W[:nx*ny, 3:] = r
    W[:nx*ny, :3] = p_
    W[nx*ny:, 3:] = p_.T

    num_batch = tf.shape(cp)[0]
    fp = tf.constant(p_[:, 1:], dtype='float32')  # [nx*ny, 2]
    fp = tf.expand_dims(fp, 0)  # [1, nx*ny, 2]
    fp = tf.tile(fp, tf.stack([num_batch, 1, 1]))  # [n, nx*ny, 2]
    W_inv = np.linalg.inv(W)
    W_inv_t = tf.constant(W_inv, dtype='float32')  # [nx*ny+3, nx*ny+3]
    W_inv_t = tf.expand_dims(W_inv_t, 0)          # [1, nx*ny+3, nx*ny+3]
    W_inv_t = tf.tile(W_inv_t, tf.stack([num_batch, 1, 1]))

    cp_pad = tf.pad(cp, [[0, 0], [0, 3], [0, 0]], "CONSTANT")
    T = tf.matmul(W_inv_t, cp_pad)
    T = tf.transpose(T, [0, 2, 1])

    return T, fp

  T, fp = _solve_system(cp, nx, ny)
  if is_points:
    output = _point_transform(T, fp, U, out_size)
  else:
    output = _transform(T, fp, U, out_size)

  return output
