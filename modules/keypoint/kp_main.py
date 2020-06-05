import torch
import torch.nn.functional as F
import numpy as np
from sys import exit as e


import modules.utils as utils
from modules.keypoint.unet import Unet
import modules.keypoint.kp_gauss as kp


def show_sample(output_imgs, oflag):
  sample_img = output_imgs.squeeze(2).detach().numpy()[6,:,:,:].squeeze()
  for i in range(oflag):
    print(np.amax(sample_img[i,:,:]), np.amin(sample_img[i,:,:]))
    utils.imshow(sample_img[i,:, :])


def generate_kp(gflag, oflag):
  unet = Unet(gflag, oflag)
  imgs = torch.load("./input/processed/0003.pt")
  lst = list(torch.split(imgs, 32, dim = 0))
  for img in lst:
    print(img.size())
    output_imgs = unet(img)
    img_shape = output_imgs.size()
    output_imgs = output_imgs.view(img_shape[0], img_shape[1], img_shape[2], -1)
    output_imgs = F.softmax(output_imgs/0.1, dim = 3)
    output_imgs = output_imgs.view(*img_shape)
    print(output_imgs.size())
    # show_sample(output_imgs, oflag)

    e()
