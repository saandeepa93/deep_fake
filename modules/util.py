import cv2
import json
from sys import exit as e
import numpy as np
import torch
from skimage.draw import circle
import matplotlib.pyplot as plt
import yaml

import torch.nn.functional as F



def draw_video_with_kp(video, kp_array, kp_size = 1, colormap = plt.get_cmap('gist_rainbow')):
  print("\nVisualizing keypoints")
  print("size required", video.shape, kp_array.shape)
  video_array = np.copy(video)
  spatial_size = np.array(video_array.shape[2:0:-1])[np.newaxis, np.newaxis]
  kp_array = spatial_size * (kp_array + 1) / 2
  num_kp = kp_array.shape[1]
  for i in range(len(video_array)):
    for kp_ind, kp in enumerate(kp_array[i]):
      rr, cc = circle(kp[1], kp[0],kp_size, shape=video_array.shape[1:3])
      video_array[i][rr, cc] = np.array(colormap(kp_ind / num_kp))[:3]
    show(video_array[i])
  return video_array


def visualize_kps(src, kp_array):
  test_img = src[:, :, 0, :, :].detach().permute(0, 2, 3, 1).numpy()
  draw_video_with_kp(test_img, kp_array['mean'][:, 0, :, :].detach().numpy())

def matrix_inverse(batch_of_matrix, eps=0):
  if eps != 0:
      init_shape = batch_of_matrix.shape
      a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
      b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
      c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
      d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

      det = a * d - b * c
      out = torch.cat([d, -b, -c, a], dim=-1)
      eps = torch.tensor(eps).type(out.type())
      out /= det.max(eps)

      return out.view(init_shape)
  else:
      b_mat = batch_of_matrix
      eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
      b_inv, _ = torch.solve(eye, b_mat)
      return b_inv

def normalize_heatmap(heatmap, norm_const):
  return heatmap/norm_const


def imread(path, m, n, gflag):
  img = cv2.imread(path)
  if (m != 0):
    img = cv2.resize(img, (m,n))
  if gflag == 1:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
  return img




def imshow(img):
  cv2.imshow("image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def show(img):
  plt.imshow(img)
  plt.show()

def get_config(config_path):
  with open(config_path) as file:
    configs = yaml.load(file, Loader = yaml.FullLoader)
  return configs


def crop_concat(upsampled, encoder):
  c = abs(upsampled.size()[3] - encoder.size()[3])//2
  encoder = F.pad(encoder, (-c, -c, -c, -c), "constant", 0)
  return torch.cat((upsampled, encoder), 1)










