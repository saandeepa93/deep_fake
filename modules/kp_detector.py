import torch
from torch import nn
import torch.nn.functional as F
from sys import exit as e

from modules.unet import Hourglass
import modules.util as util



def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size

    d = torch.linspace(-1, 1, h)
    meshx, meshy = torch.meshgrid((d, d))
    grid = torch.stack((meshy, meshx), 2)
    return grid

def normalize_heatmap(heatmap, norm_const):
  return heatmap/norm_const


def kp_mean_var(heatmap):
  mesh = make_coordinate_grid(heatmap.size()[3:], heatmap.type()).unsqueeze(0).unsqueeze(0).unsqueeze(0)
  heatmap = heatmap.unsqueeze(-1) + 1e-7

  # h_k
  mean = (heatmap * mesh).sum(dim = (3, 4))

  #sigma
  var = mesh - mean.unsqueeze(-2).unsqueeze(-2)
  #square and add another dimension for covariance
  var = torch.matmul(var.unsqueeze(-1), var.unsqueeze(-2))
  var = heatmap.unsqueeze(-1) * var
  var = var.sum(dim = (3, 4))
  kp = {"mean":mean.permute(0, 2, 1, 3), "var": var.permute(0, 2, 1, 3, 4)}
  return kp


def apply_gauss(kp, spatial_size):
  print("kp driving: ", kp["mean"].size())
  mean = kp["mean"]
  num_leading_dims = len(mean.size()) - 1
  coord_grid = make_coordinate_grid(spatial_size, mean.type)
  coord_shape = (1, ) * num_leading_dims + coord_grid.shape
  coord_grid = coord_grid.view(*coord_shape)

  # Repeat your coordinate to mean's shape
  repeats = mean.shape[:num_leading_dims] + (1, 1, 1)
  coord_grid = coord_grid.repeat(*repeats)

  mean_shape = mean.shape[:num_leading_dims] + (1, 1, 2)
  mean = mean.view(*mean_shape)

  var = kp["var"]
  inv_var = util.matrix_inverse(var)
  inv_shape = var.shape[:num_leading_dims] + (1, 1, 2, 2)
  inv_var = inv_var.view(*inv_shape)

  mean_sub = coord_grid - mean

  under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1)).squeeze(-1).squeeze(-1)
  out = torch.exp(-0.5 * under_exp)

  return out


class KeyPointDetector(nn.Module):
  def __init__(self, in_features, out_features, max_features, num_blocks, block_expansion):
    super(KeyPointDetector, self).__init__()

    self.hourglass = Hourglass(in_features, out_features, max_features, num_blocks, block_expansion)

  def forward(self, x):
    heatmap = self.hourglass(x)
    img_shape = heatmap.size()
    heatmap = F.softmax(heatmap/0.1, dim = 3)
    heatmap = heatmap.view(*img_shape)
    kp_array = kp_mean_var(heatmap)
    return kp_array
