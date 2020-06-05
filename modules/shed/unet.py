from torch import nn, optim
import torch
import torch.nn.functional as F
from sys import exit as e


class Unet(nn.Module):

  def contractor(self, in_channel, out_channel, kernel_size = 3):
    block = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(out_channel),

      torch.nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = kernel_size),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(out_channel),
    )
    return block


  def crop_concat(self, upsampled, encoder):
    c = abs(upsampled.size()[2] - encoder.size()[2])//2
    encoder = F.pad(encoder, (-c, -c, -c, -c), "constant", 0)
    return torch.cat((upsampled, encoder), 1)



  def expansion(self, in_channel, mid_channel, out_channel, kernel_size = 3):
    block = nn.Sequential(
      torch.nn.Conv2d(in_channels = in_channel, out_channels = mid_channel, kernel_size = kernel_size),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(mid_channel),

      torch.nn.Conv2d(in_channels = mid_channel, out_channels = mid_channel, kernel_size = kernel_size),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(mid_channel),

      torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1, output_padding=1)

    )
    return block

  def final(self, in_channel, mid_channel, out_channel, kernel_size = 3):
    block = nn.Sequential(
      torch.nn.Conv2d(in_channels = in_channel, out_channels = mid_channel, kernel_size = kernel_size),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(mid_channel),


      torch.nn.Conv2d(in_channels = mid_channel, out_channels = mid_channel, kernel_size = kernel_size),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(mid_channel),

      torch.nn.Conv2d(in_channels = mid_channel, out_channels = out_channel, kernel_size = kernel_size, padding = 1),
      torch.nn.ReLU(),
      # torch.nn.Softmax(dim = 1),
      torch.nn.BatchNorm2d(out_channel),
    )

    return block



  def __init__(self, in_channel, out_channel):
    super(Unet, self).__init__()
    self.encoder1 = self.contractor(in_channel = in_channel, out_channel = 64)
    self.maxpool1 = torch.nn.MaxPool2d(kernel_size = 2)
    self.encoder2 = self.contractor(64, 128)
    self.maxpool2 = torch.nn.MaxPool2d(kernel_size = 2)
    self.encoder3 = self.contractor(128, 256)
    self.maxpool3 = torch.nn.MaxPool2d(kernel_size = 2)
    self.encoder4 = self.contractor(256, 512)
    self.maxpool4 = torch.nn.MaxPool2d(kernel_size = 2)


    self.bottleneck1 = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(1024),

      torch.nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(1024),

      # torch.nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 3),
      torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
    )

    self.decoder4 = self.expansion(1024, 512, 256)
    self.decoder3 = self.expansion(512, 256, 128)
    self.decoder2 = self.expansion(256, 128, 64)

    self.final_layer = self.final(128, 64, out_channel)


  def forward(self, x):
    encode_block1 = self.encoder1(x)
    encode_pool1 = self.maxpool1(encode_block1)
    encode_block2 = self.encoder2(encode_pool1)
    encode_pool2 = self.maxpool2(encode_block2)
    encode_block3 = self.encoder3(encode_pool2)
    encode_pool3 = self.maxpool2(encode_block3)
    encode_block4 = self.encoder4(encode_pool3)
    encode_pool4 = self.maxpool4(encode_block4)

    bottleneck1 = self.bottleneck1(encode_pool4)

    cat_layer4 = self.crop_concat(bottleneck1, encode_block4)
    decode_block3 = self.decoder4(cat_layer4)
    cat_layer3 = self.crop_concat(decode_block3, encode_block3)
    decode_block2 = self.decoder3(cat_layer3)
    cat_layer2 = self.crop_concat(decode_block2, encode_block2)
    decode_block1 = self.decoder2(cat_layer2)
    cat_layer1 = self.crop_concat(decode_block1, encode_block1)

    final_block = self.final_layer(cat_layer1)

    # softmax_block = torch.sigmoid(final_block/0.1)
    softmax_block = F.softmax(final_block/0.1, dim = 3)

    return encode_pool1

