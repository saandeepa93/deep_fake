import cv2
import numpy as np
import os
from sys import exit as e
from tqdm import tqdm
import torch

import modules.util as util

def read_data(configs):
  """Extract Video frames from the input path and store in a tensor of format (N, C, D, H, W). create 2 pairs of tensor each for 1-channel and 3-channel.

  Arguments:
      input_path {string} -- Input path to fetch the videos/images
      gflag {int} -- Channels of the input path (Typically 1). Also saves 3 channel tensors
  """
  input_path = configs['paths']['input']
  gflag = configs['image']['in_channel']
  pro_path = configs['paths']['processed']
  img_size = configs['image']['img_size']

  source_lst = []
  source_lst_rgb = []
  driving_lst = []
  driving_lst_rgb = []
  for folder in os.listdir(input_path):
    if folder.isnumeric():
      folder_path = os.path.join(input_path, folder)
      for files in tqdm(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, files)
        if os.path.splitext(fpath)[1] == '.jpg':
          source_lst.append(util.imread(fpath, img_size, img_size, gflag))
          source_lst_rgb.append(util.imread(fpath, img_size, img_size, 3))
      img_arr = torch.from_numpy(np.array(source_lst)).unsqueeze(1).unsqueeze(2)
      img_arr_rgb = torch.from_numpy(np.array(source_lst_rgb)).unsqueeze(1).permute(0, 4, 1, 2, 3)
      print("source shape:", img_arr.shape, img_arr_rgb.size())
      save_path = os.path.join(pro_path, "source.pt")
      save_path_rgb = os.path.join(pro_path, "source_rgb.pt")
      torch.save(img_arr, save_path)
      torch.save(img_arr, save_path_rgb)

      cap = cv2.VideoCapture(os.path.join(input_path, 'facial_expression.mov'))
      ret = True
      while ret:
        ret, frame = cap.read()
        if ret:
          frame_rgb = cv2.resize(frame, (img_size, img_size))
          frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (img_size, img_size))
          frame = cv2.normalize(frame, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
          driving_lst.append(frame)
          driving_lst_rgb.append(frame_rgb)
      driving_arr = torch.from_numpy(np.array(driving_lst)).unsqueeze(1).unsqueeze(2)
      driving_arr_rgb = torch.from_numpy(np.array(driving_lst_rgb)).unsqueeze(1).permute(0, 4, 1, 2, 3)
      save_path = os.path.join(pro_path, 'driving.pt')
      save_path_rgb = os.path.join(pro_path, 'driving_rgb.pt')
      torch.save(driving_arr, save_path)
      torch.save(driving_arr, save_path_rgb)
      print("driving shape:", driving_arr.size(), driving_arr_rgb.size())


      e()
