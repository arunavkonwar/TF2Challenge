#installing Tensorflow 2.0
# pip install tensorflow-gpu==2.0.0-alpha0

from __future__ import print_function
from io import BytesIO
import numpy as np
import PIL

import matplotlib.pylab as pl

from google.colab import files

import tensorflow as tf
from tensorflow.contrib import slim

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import moviepy.editor as mvp

from IPython.display import clear_output, Image, display


from lucid.modelzoo import vision_models
from lucid.misc.io import load, save, show
from lucid.optvis import objectives, transform
from lucid.optvis.param import image, image_sample
from lucid.misc.tfutil import create_session

model = vision_models.InceptionV1()
model.load_graphdef()

from math import ceil

def checkerboard(h, w=None, channels=3, tiles=4, fg=.95, bg=.6):
  """Create a shape (w,h,1) array tiled with a checkerboard pattern."""
  w = w or h
  square_size = [ceil(float(d / tiles) / 2) for d in [h, w]]
  board = [[fg, bg] * tiles, [bg, fg] * tiles] * tiles
  scaled = np.kron(board, np.ones(square_size))[:w, :h]
  return np.dstack([scaled]*channels)


for fg, bg in [(1., 0.), (.95, .6)]:
  print("\nColors: {}".format((fg, bg)))
  show([checkerboard(128, fg=fg, bg=bg, tiles=i) 
        for i in (1, 2, 4, 8)])






def linear2gamma(a):
  return a**(1.0/2.2)

def gamma2linear(a):
  return a**2.2


def composite_alpha_onto_backgrounds(rgba,
                                     input_encoding='gamma',
                                     output_encoding='gamma'):
  if input_encoding == 'gamma':
    rgba = rgba.copy()
    rgba[..., :3] = gamma2linear(rgba[..., :3])
    
  h, w = rgba.shape[:2]
  
  # Backgrounds
  black = np.zeros((h, w, 3), np.float32)
  white = np.ones((h, w, 3), np.float32)
  grid = checkerboard(h, w, 3, tiles=8)
  
  # Collapse transparency onto backgrounds
  rgb, a = rgba[...,:3], rgba[...,3:]
  vis = [background*(1.0-a) + rgb*a for background in [black, white, grid]]
  vis.append(white*a) # show just the alpha channel separately
  
  # Reshape into 2x2 grid
  vis = np.float32(vis).reshape(2, 2, h, w, 3)
  vis = np.vstack(map(np.hstack, vis))
  if output_encoding == 'gamma':
    vis = linear2gamma(vis)
  return vis


rgba_image = load("https://storage.googleapis.com/tensorflow-lucid/notebooks/rgba/rgba-example.png")
show(composite_alpha_onto_backgrounds(rgba_image))




def render(obj_str, iter_n=1000):
  sess = create_session()

  def T(layer):
    return sess.graph.get_tensor_by_name("import/%s:0"%layer)

  w, h = 320, 320
  t_image = image(h, w, decorrelate=True, fft=True, alpha=True)
  t_rgb = t_image[...,:3]
  t_alpha = t_image[...,3:]
  t_bg = image_sample([1, h, w, 3], sd=0.2, decay_power=1.5)
  
  t_composed = t_bg*(1.0-t_alpha) + t_rgb*t_alpha
  t_composed = tf.concat([t_composed, t_alpha], -1)
  t_crop = transform.random_scale([0.6, 0.7, 0.8, 0.9, 1.0, 1.1])(t_composed)
  t_crop = tf.random_crop(t_crop, [1, 160, 160, 4])
  t_crop_rgb, t_crop_alpha = t_crop[..., :3], t_crop[..., 3:]
  t_crop_rgb = linear2gamma(t_crop_rgb)
  model.import_graph(t_crop_rgb)

  obj = objectives.as_objective(obj_str)(T)
  t_alpha_mean_crop = tf.reduce_mean(t_crop_alpha)
  t_alpha_mean_full = tf.reduce_mean(t_alpha)
  tf.losses.add_loss(-obj*(1.0-t_alpha_mean_full)*0.5)
  tf.losses.add_loss(-obj*(1.0-t_alpha_mean_crop))


  t_lr = tf.constant(0.01)
  t_loss = tf.losses.get_total_loss()
  trainer = tf.train.AdamOptimizer(t_lr)
  train_op = trainer.minimize(t_loss)

  init_op = tf.global_variables_initializer()
  init_op.run()
  log = []

  out_name = obj_str.replace(':', '_')
  writer = FFMPEG_VideoWriter(out_name+'.mp4', (w*2, h*2), 60.0)
  with writer:
    for i in range(iter_n):
      _, loss, img = sess.run([train_op, t_loss, t_image], {t_lr: 0.03})
      vis = composite_alpha_onto_backgrounds(img[0], 'linear', 'gamma')
      writer.write_frame(np.uint8(vis*255.0))
      log.append(loss)
      if i%100 == 0:
        clear_output()
        print(len(log), loss)
        vis = t_composed.eval()[...,:3]
        vis = linear2gamma(vis)
        show(vis)
  
  img = t_image.eval()[0]
  vis = composite_alpha_onto_backgrounds(img, 'linear', 'gamma')
  img[..., :3] = linear2gamma(img[..., :3])
  show(vis)
  save(img, out_name+'.png')
  
  alpha = img[...,3]
  alpha = np.dstack([alpha]*3)
  joined = np.hstack([img[...,:3], alpha])
  save(joined, out_name+'.jpg', quality=90)

  return log


loss_log = render('mixed4d_3x3_bottleneck_pre_relu:139')

pl.plot(loss_log)
pl.xlabel('Step')
pl.ylabel('Loss');

mvp.ipython_display('mixed4d_3x3_bottleneck_pre_relu_139.mp4', height=320*2)

render('mixed4b_pool_reduce_pre_relu:16')