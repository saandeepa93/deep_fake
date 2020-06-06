from torch import nn, optim
import torch
import torch.nn.functional as F
from sys import exit as e


class Unet(nn.Module):
  def contractor(self, in_channel, out_channel, kernel_size = (1, 3, 3), padding = (0, 1, 1)):
    block = torch.nn.Sequential(
      torch.nn.Conv3d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size,\
        padding = padding),
      torch.nn.BatchNorm3d(out_channel, affine = True),
      torch.nn.ReLU(),
      torch.nn.AvgPool3d(kernel_size = (1, 2, 2)),

    )
    return block


  def crop_concat(self, upsampled, encoder):
    c = abs(upsampled.size()[3] - encoder.size()[3])//2
    encoder = F.pad(encoder, (-c, -c, -c, -c), "constant", 0)
    return torch.cat((upsampled, encoder), 1)


  def expansion(self, in_channel, out_channel, kernel_size = (1, 3, 3), padding = (0, 1, 1)):
    block = nn.Sequential(
      torch.nn.Conv3d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size,\
        padding = padding),
      torch.nn.BatchNorm3d(out_channel),
      torch.nn.ReLU(),
    )
    return block




  def __init__(self, in_channel, out_channel):
    super(Unet, self).__init__()
    self.encoder1 = self.contractor(in_channel = in_channel, out_channel = 64)
    self.encoder2 = self.contractor(64, 128)
    self.encoder3 = self.contractor(128, 256)
    self.encoder4 = self.contractor(256, 512)
    self.encoder5 = self.contractor(512, 512)

    self.decoder5 = self.expansion(512, 512)
    self.decoder4 = self.expansion(1024, 256)
    self.decoder3 = self.expansion(512, 128)
    self.decoder2 = self.expansion(256, 64)
    self.decoder1 = self.expansion(128, 32)

    self.final = nn.Conv3d(in_channels = 32, out_channels = out_channel, kernel_size= (1, 3, 3),\
      padding = (0, 1, 1))




  def forward(self, x):
    encode_block1 = self.encoder1(x)
    encode_block2 = self.encoder2(encode_block1)
    encode_block3 = self.encoder3(encode_block2)
    encode_block4 = self.encoder4(encode_block3)
    encode_block5 = self.encoder5(encode_block4)

    decode_block5 = F.interpolate(encode_block5, scale_factor = (1, 2, 2))
    decode_block5 = self.decoder5(decode_block5)
    cat_layer5 = self.crop_concat(decode_block5, encode_block4)

    decode_block4 = F.interpolate(cat_layer5, scale_factor = (1, 2, 2))
    decode_block4 = self.decoder4(decode_block4)
    cat_layer4 = self.crop_concat(decode_block4, encode_block3)

    decode_block3 = F.interpolate(cat_layer4, scale_factor = (1, 2, 2))
    decode_block3 = self.decoder3(decode_block3)
    cat_layer3 = self.crop_concat(decode_block3, encode_block2)

    decode_block2 = F.interpolate(cat_layer3, scale_factor = (1, 2, 2))
    decode_block2 = self.decoder2(decode_block2)
    cat_layer2 = self.crop_concat(decode_block2, encode_block1)

    decode_block1 = F.interpolate(cat_layer2, scale_factor = (1, 2, 2))
    decode_block1 = self.decoder1(decode_block1)

    final_layer = self.final(decode_block1)

    return final_layer

