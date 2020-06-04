import torch
from torch import nn, optim
from sys import exit as e
import numpy as np

import modules.utils as utils
from modules.keypoint.unet import Unet
import modules.keypoint.kp_gauss as kp
from modules.shed.conv3d import Convolution

def main():
  img_path = './artifacts/0035_2m_-15P_10V_10H.jpg'
  # img_path = './input/download.jpeg'
  # img_path = './input/traffic.jpg'

  gflag = 1
  oflag = 10
  img = utils.imread(img_path, 572, 572, gflag)
  img = utils.reshape_tensor(torch.from_numpy(img), gflag)

  img = img.view(1, 1, 572, 572)
  conv = Convolution(gflag, oflag)
  print(img.size())
  output_img = conv(img.float()).detach().squeeze().permute(1, 2, 0)
  print(output_img.size())
  for i in range(oflag):
    print(torch.max(output_img[:, :, i]), torch.min(output_img[:, :, i]))
    utils.imshow(output_img[:, :, i].numpy())
  e()
  # output_img = output_img.numpy().squeeze().reshape(286, 286, 10)
  # utils.imshow(output_img.numpy())

  unet = Unet(gflag, oflag)
  output_img = unet(img).detach().squeeze().permute(1, 2, 0)
  # kp.kp_mean_var(output_img)
  # utils.imshow(output_img.numpy())
  for i in range(oflag):
    print(np.amax(output_img[:, : , i].numpy()), np.amin(output_img[:, :, i].numpy()))
    # utils.imshow(output_img[:, :, i].numpy())



if __name__ == '__main__':
  main()