"""DeepLab v3 models based on slim library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2

def resnet_v2_light(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_light'):
  """ResNet-light model of AIlab. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2.resnet_v2_block('block1', base_depth=16, num_units=2, stride=2),
      resnet_v2.resnet_v2_block('block2', base_depth=32, num_units=3, stride=2),
      resnet_v2.resnet_v2_block('block3', base_depth=64, num_units=3, stride=2),
      resnet_v2.resnet_v2_block('block4', base_depth=128, num_units=2, stride=1),
  ]

  return resnet_v2.resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True,
                   reuse=reuse, scope=scope)

resnet_v2_light.default_image_size = resnet_v2.resnet_v2.default_image_size