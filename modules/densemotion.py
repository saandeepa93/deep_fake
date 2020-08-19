import torch
from torch import nn

from modules.movement_embedding import MovementEmbeddingModule

class DenseMotionModule(nn.Module):
  def __init__(self, in_features, out_features, kp_variance, block_expansions,\
    num_blocks, norm_const, use_heatmaps, use_deformed_source_image, heatmap_type):
    super(DenseMotionModule, self).__init__()

    self.mask_embedding = MovementEmbeddingModule(in_features, out_features, kp_variance,\
      norm_const, use_heatmaps, use_deformed_source_image, heatmap_type, add_bg_feature_maps = True)

  def forward(self, source_img, kp_source, kp_driving):
    predictions = self.mask_embedding(source_img, kp_source, kp_driving)
    print("predictions: ", predictions.size())