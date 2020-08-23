import torch
from torch import nn
from sys import exit as e

from modules.unet import Encoder
from modules.densemotion import DenseMotionModule

class Generator(nn.Module):
  def __init__(self, in_features, out_features, max_features, num_blocks, block_expansion,\
    kp_variance, norm_const, num_group_blocks, use_mask, use_correction, use_heatmaps, use_deformed_source_image, heatmap_type):
    super(Generator, self).__init__()

    self.appearance_encoder = Encoder(in_features, out_features, max_features,\
      num_blocks, block_expansion)
    self.densemotion = DenseMotionModule(in_features, out_features, max_features, kp_variance,\
      block_expansion, num_blocks, norm_const, num_group_blocks, use_mask, use_correction, use_heatmaps, use_deformed_source_image, heatmap_type)


  def forward(self, x, kp_source_mean, kp_driving_mean):
    appearance_skips = self.appearance_encoder(x)
    deformation_absolute = self.densemotion(x, kp_source_mean, kp_driving_mean)

