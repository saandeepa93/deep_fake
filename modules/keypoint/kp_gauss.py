import torch
from torch import nn
from sys import exit as e


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    d = torch.linspace(-1, 1, h)
    meshx, meshy = torch.meshgrid((d, d))
    grid = torch.stack((meshy, meshx), 2)
    return grid



def kp_mean_var(heatmap):
  mesh = make_coordinate_grid(heatmap.size()[3:], heatmap.type()).unsqueeze(0).unsqueeze(0).unsqueeze(0)
  heatmap = heatmap.unsqueeze(-1) + 1e-7
  mean = (heatmap * mesh).sum(dim = (3, 4))

  var = mesh - mean.unsqueeze(-2).unsqueeze(-2)
  #square and add another dimension for covariance
  var = torch.matmul(var.unsqueeze(-1), var.unsqueeze(-2))
  var = heatmap.unsqueeze(-1) * var
  var = var.sum(dim = (3, 4))
  kp = {"mean":mean.permute(0, 2, 1, 3), "var": var.permute(0, 2, 1, 3, 4)}
  print(f"heatmap: {heatmap.size()}, mesh: {mesh.size()}, mean: {mean.size()},\
    variance: {var.size()}")




  e()

