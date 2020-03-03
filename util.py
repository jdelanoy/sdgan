import argparse

import tensorflow as tf


# Decode bool argument
def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


# Decode PNG to [-1, 1] float
def decode_png_observation(png_fp):
  png_bin = tf.read_file(png_fp)
  png = tf.image.decode_image(png_bin, channels=3)
  # assertion=tf.assert_equal(tf.shape(png)[-1],3,message="image  "+png_fp+" should have at least 3 channels")
  # with tf.control_dependencies([assertion]):
  #   png=tf.identity(png)
  png = tf.cast(png, tf.float32)
  png /= 256.
  png *= 2.
  png -= 1.

  return png


# Encode [-1, 1] float to uint8
def encode_png_observation(png, name=None):
  png += 1.
  png /= 2.
  png *= 256.
  png = tf.clip_by_value(png, 0., 255.)

  return tf.cast(png, tf.uint8, name=name)
