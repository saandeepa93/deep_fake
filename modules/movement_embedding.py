import torch
from torch import nn
import torch.nn.functional as F
from sys import exit as e


from modules.kp_detector import apply_gauss, normalize_heatmap
import modules.kp_detector as kp

class MovementEmbeddingModule(nn.Module):
  def __init__(self, in_features, out_features, kp_variance, norm_const, use_heatmaps = True,\
    use_deformed_source_image = False, heatmap_type = "gaussian", add_bg_feature_maps = False,  use_difference=False):
    super(MovementEmbeddingModule, self).__init__()

    self.use_heatmaps = use_heatmaps
    self.norm_const = norm_const
    self.heatmap_type = heatmap_type
    self.add_bg_feat = add_bg_feature_maps
    self.use_difference = use_difference
    self.use_deformed_source_image = use_deformed_source_image

  def forward(self, source_img, kp_source, kp_driving):
    inputs = []
    b, _, _, h, w = source_img.size()
    _, d, num_kp, _ = kp_driving["mean"].size()

    if self.use_heatmaps:
      heatmap = normalize_heatmap(apply_gauss(kp_driving, source_img.size()[3:5]), self.norm_const)
      if self.heatmap_type == "difference":
        heatmap_source = normalize_heatmap(apply_gauss(kp_source, source_img.size()[3:5]), self.norm_const)
        heatmap -= heatmap_source


      if self.add_bg_feat:
        zeros = torch.zeros(b, 1, 1, h, w).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim = 2)

      heatmap = heatmap.unsqueeze(3)
      inputs.append(heatmap)



      num_kp += self.add_bg_feat

      if self.use_difference or self.use_deformed_source_image:
        kp_diff = (kp_driving["mean"] - kp_source["mean"])
        if self.add_bg_feat:
          zeros = torch.zeros(b, 1, 1, 2).type(kp_diff.type())
          kp_diff = torch.cat([zeros, kp_diff], dim = 2)
        kp_diff = kp_diff.view(b, 1, num_kp, 2, 1, 1).repeat(1, 1, 1, 1, h, w)
        print("kp diff: ", kp_diff.size())

      if self.use_difference:
        inputs.append(kp_diff)

    if self.use_deformed_source_image:
      appearance_repeat = source_img.unsqueeze(1).unsqueeze(1).repeat(1, d, num_kp, 1, 1, 1, 1)
      appearance_repeat = appearance_repeat.view(b * d* num_kp, -1, h, w)
      deform_approx = kp_diff.view(b*d*num_kp, -1, h, w).permute(0, 2, 3, 1)
      coordinate_grid = kp.make_coordinate_grid((h, w), deform_approx.type()).unsqueeze(0)
      deform_approx += coordinate_grid
      appearance_approx_deform = F.grid_sample(appearance_repeat, deform_approx, align_corners = True)
      appearance_approx_deform = appearance_approx_deform.view(b, d, num_kp, -1, h, w)
      inputs.append(appearance_approx_deform)

    movement_encoding = torch.cat(inputs, dim = 3).view(b, d, -1, h, w)
    return movement_encoding.permute(0, 2, 1, 3, 4)










