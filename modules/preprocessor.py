import cv2
import numpy as np
import os
from sys import exit as e
from tqdm import tqdm
import torch

import modules.utils as utils

def read_data(input_path, gflag):
  img_lst = []
  pro_path = utils.get_config("processed_path")
  img_size = utils.get_config("img_size")
  for folder in os.listdir(input_path):
    if folder.isnumeric():
      folder_path = os.path.join(input_path, folder)
      for files in tqdm(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, files)
        if os.path.splitext(fpath)[1] == '.jpg':
          img_lst.append(utils.imread(fpath, img_size, img_size, gflag))
      img_arr = torch.from_numpy(np.array(img_lst)).unsqueeze(1).unsqueeze(2)
      print(img_arr.shape)
      save_path = os.path.join(pro_path, folder+".pt")
      torch.save(img_arr, save_path)

      e()
