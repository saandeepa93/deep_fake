import torch
from torch.utils.data import DataLoader
from sys import exit as e

from modules.keypoint.kp_main import generate_kp
from modules.dataset import DataClass
import modules.util as util

def train_data(configs):
  root_folder = configs['paths']['input']
  img_size = configs['hypers']['size']
  ext = configs['params']['ext']
  b_size = configs['hypers']['batch_size']

  dataset = DataClass(root_folder, img_size, ext)
  dataloader = DataLoader(dataset, batch_size = b_size, shuffle = False)


  for b, img in enumerate(dataloader, 0):
    img_size = img.size()
    img = img.view(-1, img_size[2], img_size[3], img_size[4], img_size[5]).permute(0, 2, 1, 3, 4)
    output = generate_kp(configs, img)
