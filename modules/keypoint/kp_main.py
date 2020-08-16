import torch
import torch.nn.functional as F
import numpy as np
from sys import exit as e


import modules.util as util
from modules.keypoint.unet import Unet
import modules.keypoint.kp_gauss as kp


def show_sample(output_imgs, oflag):
  sample_img = output_imgs.detach().numpy()[6,:,0,:,:].squeeze()
  dest_img = output_imgs.detach().numpy()[6,:,1,:,:].squeeze()
  for i in range(oflag):
    print(np.amax(sample_img[i,:,:]), np.amin(sample_img[i,:,:]))
    util.imshow(sample_img[i,:, :])
    util.imshow(dest_img[i,:, :])


def generate_kp(configs, src):
  gflag = configs['hypers']['in_channel']
  oflag = configs['hypers']['out_channel']
  unet = Unet(gflag, oflag, 0)
  output_imgs = unet(src)
  img_shape = output_imgs.size()
  output_imgs = output_imgs.view(img_shape[0], img_shape[1], img_shape[2], -1)
  output_imgs = F.softmax(output_imgs/0.1, dim = 3)
  output_imgs = output_imgs.view(*img_shape)
  kp_array = kp.kp_mean_var(output_imgs)
  heatmap_gauss = util.normalize_heatmap(kp.apply_gauss(kp_array, src.size()[3:5]), configs['params']['norm_const'])
  # util.visualize_kps(src, kp_array)

  # calculating delta heatmap of source and driving and adding background 0s
  inputs = []
  heatmap_diff = (heatmap_gauss[:, 0] - heatmap_gauss[:, 1]).unsqueeze(1)
  b, d, _, h, w = heatmap_diff.size()
  _, _, num_kp, _ = kp_array["mean"].size()
  zeros = torch.zeros(b, 1, 1, h, w).type(heatmap_diff.type())
  heatmap_diff = torch.cat([zeros, heatmap_diff], dim = 2)
  heatmap_diff = heatmap_diff.unsqueeze(3)
  inputs.append(heatmap_diff)

  kp_diff = (kp_array["mean"][:, 0] - kp_array["mean"][:, 1]).unsqueeze(1)

  #caluculating delta mean of source and driving video and adding background 0s
  num_kp += 1
  zeros = torch.zeros(b, 1, 1, 2).type(kp_diff.type())
  kp_diff = torch.cat([zeros, kp_diff], dim = 2)
  kp_diff = kp_diff.view(b, 1, num_kp, 2, 1, 1).repeat(1, 1, 1, 1, h, w)

  #Grid sampling source image and delta mean to calculate deformations
  source_img = src[:, :, 0].unsqueeze(2)
  appearance_repeat = source_img.unsqueeze(1).unsqueeze(1).repeat(1, d, num_kp, 1, 1, 1, 1)
  appearance_repeat = appearance_repeat.view(b * d* num_kp, -1, h, w)
  deform_approx = kp_diff.view(b*d*num_kp, -1, h, w).permute(0, 2, 3, 1)
  coordinate_grid = kp.make_coordinate_grid((h, w), deform_approx.type()).unsqueeze(0)
  deform_approx += coordinate_grid
  appearance_approx_deform = F.grid_sample(appearance_repeat, deform_approx, align_corners = True)
  appearance_approx_deform = appearance_approx_deform.view(b, d, num_kp, -1, h, w)

  inputs.append(appearance_approx_deform)

  movement_encoding = torch.cat(inputs, dim = 3).view(b, d, -1, h, w)

  print("kp diff: ", kp_diff.size())
  print("heatmap diff: ", heatmap_diff.size())
  print("appearance repeat: ", appearance_repeat.size())
  print("deform_approx: ", deform_approx.size())
  print("coordinate grid: ", coordinate_grid.size())
  print("appearance approx deform: ", appearance_approx_deform.size())
  print("movement encoding: ", movement_encoding.size())
  e()


