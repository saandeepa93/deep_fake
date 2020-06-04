import torch
from torch import nn
from sys import exit as e


def get_grid_coordinates(heatmap_size, heatmap_type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = heatmap_size[:2]
    x = torch.arange(w).type(heatmap_type)
    y = torch.arange(h).type(heatmap_type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    return meshed


def kp_mean_var(heatmap):
  mesh = get_grid_coordinates(heatmap.size(), heatmap.type())
  heatmap = heatmap.unsqueeze(0).permute(3, 1, 2, 0)
  mesh = mesh.unsqueeze(0)

  mean = (heatmap * mesh).sum(dim = (1, 2))
  print(mean)
