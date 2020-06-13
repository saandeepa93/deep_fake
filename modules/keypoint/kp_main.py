import torch
import torch.nn.functional as F
import numpy as np
from sys import exit as e


import modules.utils as utils
from modules.keypoint.unet import Unet
import modules.keypoint.kp_gauss as kp


def show_sample(output_imgs, oflag):
  sample_img = output_imgs.detach().numpy()[6,:,0,:,:].squeeze()
  dest_img = output_imgs.detach().numpy()[6,:,1,:,:].squeeze()
  for i in range(oflag):
    print(np.amax(sample_img[i,:,:]), np.amin(sample_img[i,:,:]))
    utils.imshow(sample_img[i,:, :])
    utils.imshow(dest_img[i,:, :])


def generate_kp(gflag, oflag):
  unet = Unet(gflag, oflag, 0)
  source = torch.load("./input/processed/source.pt")
  # driving = torch.load("./input/processed/driving.pt")
  # driving = driving[30:135,:,:,:,:]
  # x = torch.cat((source, driving), dim = 2)
  x = torch.cat((source[0:32, :, :, :, :], source[64:96, :, :, :, :]), dim = 2)
  x = list(torch.split(x, 32, dim = 0))
  for src in x:
    output_imgs = unet(src)
    # for enc in output_imgs:
    #   print(enc.size())
    print(output_imgs.size())
    show_sample(output_imgs, oflag)
    e()
    img_shape = output_imgs.size()
    output_imgs = output_imgs.view(img_shape[0], img_shape[1], img_shape[2], -1)
    output_imgs = F.softmax(output_imgs/0.1, dim = 3)
    output_imgs = output_imgs.view(*img_shape)
    kp.kp_mean_var(output_imgs)
    e()
