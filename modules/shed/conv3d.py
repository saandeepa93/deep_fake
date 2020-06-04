import torch
from torch import nn


import modules.shed.utils as utils


class Convolution(nn.Module):
  def __init__(self, in_chan, out_chan):
    super(Convolution, self).__init__()

    self.model = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels = in_chan, out_channels = out_chan, kernel_size = 3),
      torch.nn.ReLU(),
      # torch.nn.BatchNorm2d(out_chan, affine = True),
      torch.nn.AvgPool2d(2)
    )

    # self.model = nn.Sequential(
    #   nn.Conv3d(in_channels = in_chan, out_channels = 64, kernel_size = (1, 3, 3),\
    #     stride = (1, 1, 1), padding = (0, 1, 1)),
    #   nn.BatchNorm3d(64),
    #   nn.AvgPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2), padding = 0),

    # )


  def forward(self, x):
    print("input size", x.size())
    return self.model(x)