import torch
from torch import nn
from torch.nn import functional as F
from sys import exit as e

from modules.movement_embedding import MovementEmbeddingModule
from modules.unet import Hourglass, SameBlock3d
from modules.kp_detector import make_coordinate_grid
import modules.util as util

class DenseMotionModule(nn.Module):
  def __init__(self, in_features, out_features, max_features, kp_variance, block_expansion,\
    num_blocks, norm_const, num_group_blocks, use_mask, use_correction, use_heatmaps,\
      use_deformed_source_image, heatmap_type, bg_init = 2):
    super(DenseMotionModule, self).__init__()

    self.use_mask = use_mask
    self.use_correction = use_correction
    self.num_kp = out_features

    self.mask_embedding = MovementEmbeddingModule(in_features, out_features, kp_variance,\
      norm_const, use_heatmaps, use_deformed_source_image, heatmap_type, add_bg_feature_maps = True)

    self.difference_embedding = MovementEmbeddingModule(in_features = in_features, out_features = out_features,\
      kp_variance = kp_variance, norm_const = norm_const, add_bg_feature_maps=True, use_difference=True, use_heatmaps=False,\
        use_deformed_source_image = False)



    same_blocks = []
    for i in range(num_group_blocks):
      same_blocks.append(SameBlock3d(self.mask_embedding.out_channels, self.mask_embedding.out_channels,\
        groups = out_features+1, kernel_size=(1, 1, 1), padding = (0, 0, 0)))

    self.same_blocks = nn.ModuleList(same_blocks)

    ## question - why?
    self.hourglass = Hourglass(in_features = self.mask_embedding.out_channels,\
      out_features = (out_features + 1) * use_mask + (2 * use_correction), max_features = max_features,\
        num_blocks = num_blocks, block_expansion = block_expansion)

    self.hourglass.decoder.last_conv.weight.data.zero_()
    bias_init = ([bg_init] + [0] * out_features) + ([0, 0] * use_correction)
    self.hourglass.decoder.last_conv.bias.data.copy_(torch.tensor(bias_init, dtype = torch.float))


  def forward(self, source_img, kp_source, kp_driving):
    predictions = self.mask_embedding(source_img, kp_source, kp_driving)


    for block in self.same_blocks:
      predictions = block(predictions)
      predictions = F.leaky_relu(predictions)
    predictions = self.hourglass(predictions)

    b, _, _, h, w = predictions.size()
    print("predictions: ", predictions.size())
    if self.use_mask:
      mask = predictions[:, :self.num_kp+1]
      mask = F.softmax(mask, dim = 1)
      mask = mask.unsqueeze(2)
      print("mask: ", mask.size())

      difference_embedding  = self.difference_embedding(source_img, kp_source, kp_driving)
      difference_embedding = difference_embedding.view(b, self.num_kp+1, 2, -1, h, w)
      print("difference embedding: ", difference_embedding.size())

      deformation_relative = (difference_embedding * mask).sum(dim = 1)
      print("deformation relative: ", deformation_relative.size())

    else:
      deformation_relative = 0

    if self.use_correction:
      correction = predictions[:, -2:]
      print("correction: ", correction.size())
    else:
      correction = 0


    coordinate_grid = make_coordinate_grid((h, w), deformation_relative.type())
    coordinate_grid = coordinate_grid.unsqueeze(0).unsqueeze(0)



