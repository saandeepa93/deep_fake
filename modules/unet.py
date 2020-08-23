import torch
from torch import nn
from sys import exit as e
import torch.nn.functional as F

import modules.util as util


class DownBlock(nn.Module):
  def __init__(self, in_channel, out_channel, kernel_size, padding):
    super(DownBlock, self).__init__()
    self.block = torch.nn.Sequential(
      torch.nn.Conv3d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size,\
        padding = padding),
      torch.nn.BatchNorm3d(out_channel, affine = True),
      torch.nn.ReLU(),
      torch.nn.AvgPool3d(kernel_size = (1, 2, 2)),
    )

  def forward(self, x):
    return self.block(x)



class UpBlock(nn.Module):
  def __init__(self, in_channel, out_channel, kernel_size, padding):
    super(UpBlock, self).__init__()
    self.block = nn.Sequential(
      torch.nn.Conv3d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size,\
        padding = padding),
      torch.nn.BatchNorm3d(out_channel),
      torch.nn.ReLU(),
    )

  def forward(self, x):
    return self.block(x)


class Encoder(nn.Module):
  def __init__(self, in_features, out_features, max_features, num_blocks, block_expansion):
    super(Encoder, self).__init__()
    downblocks = []
    kernel_size = (1, 3, 3)
    padding = (0, 1, 1)
    for i in range(num_blocks):
      downblocks.append(DownBlock(in_features if i==0 else min(max_features, block_expansion * (2**i))\
      , min(max_features, block_expansion * (2**(i+1))), kernel_size, padding))

    self.downblocks = nn.ModuleList(downblocks)


  def forward(self, x):
    outs = [x]
    for block in self.downblocks:
      outs.append(block(outs[-1]))
    return outs


class Decoder(nn.Module):
  def __init__(self, in_features, out_features, max_features, num_blocks, block_expansion):
    super(Decoder, self).__init__()

    upblocks = []
    kernel_size = (1, 3, 3)
    padding = (0, 1, 1)

    for i in range(num_blocks-1, -1, -1):
      upblocks.append(UpBlock((1 if i==num_blocks-1 else 2) * min(max_features, block_expansion * (2**(i+1)))\
        , min(max_features, block_expansion * (2**i)), kernel_size, padding))


    self.upblocks = nn.ModuleList(upblocks)
    self.last_conv = nn.Conv3d(in_channels = 32, out_channels = out_features, kernel_size= (1, 3, 3),\
      padding = (0, 1, 1))


  def forward(self, x):
    out = x.pop()
    for block in self.upblocks:
      out = F.interpolate(out, scale_factor = (1, 2, 2))
      out = block(out)
      if len(x) != 1:
        out = util.crop_concat(out, x.pop())
      else:
        continue
    out = self.last_conv(out)
    return out


class Hourglass(nn.Module):
  def __init__(self, in_features, out_features, max_features, num_blocks, block_expansion):
    super(Hourglass, self).__init__()

    self.encoder = Encoder(in_features, out_features, max_features, num_blocks, block_expansion)
    self.decoder = Decoder(in_features, out_features, max_features, num_blocks, block_expansion)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x


class SameBlock3d(nn.Module):
  def __init__(self, in_features, out_features, kernel_size, padding, groups):
    super(SameBlock3d, self).__init__()
    self.conv3d = nn.Conv3d(in_channels = in_features, out_channels = out_features,\
      kernel_size = kernel_size, padding = padding, groups = groups)

    self.batchnorm = nn.BatchNorm3d(out_features, affine = True)


  def forward(self, x):
    x = self.conv3d(x)
    x = self.batchnorm(x)
    return F.relu(x)








