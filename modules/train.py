import torch
from torch.utils.data import DataLoader
from sys import exit as e

from modules.dataset import DataClass
import modules.util as util
from modules.kp_detector import KeyPointDetector

def train_data(configs):
  root_folder = configs['paths']['input']
  img_size = configs['hypers']['size']
  ext = configs['params']['ext']
  b_size = configs['hypers']['batch_size']


  dataset = DataClass(root_folder, img_size, ext)
  dataloader = DataLoader(dataset, batch_size = b_size, shuffle = False)

  in_features = configs["hypers"]["in_channel"]
  out_features = configs["hypers"]["out_channel"]
  max_features = configs["hypers"]["max_features"]
  num_block = configs["hypers"]["num_blocks"]
  block_expansion = configs["hypers"]["block_expansion"]

  kp_detector = KeyPointDetector(in_features, out_features, max_features, num_block, block_expansion)

  for b, img in enumerate(dataloader, 0):
    img_size = img.size()
    img = img.view(-1, img_size[2], img_size[3], img_size[4], img_size[5]).permute(0, 2, 1, 3, 4)
    kp = kp_detector(img)
    print(kp["mean"].size(), kp["var"].size())
    e()
    # output = generate_kp(configs, img)

