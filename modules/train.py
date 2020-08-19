import torch
from torch.utils.data import DataLoader
from sys import exit as e

from modules.dataset import DataClass
import modules.util as util
from modules.kp_detector import KeyPointDetector
from modules.generator import Generator



def split_kp(kp_joined, detach=False):
    if detach:
        kp_video = {k: v[:, 1:].detach() for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1].detach() for k, v in kp_joined.items()}
    else:
        kp_video = {k: v[:, 1:] for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1] for k, v in kp_joined.items()}
    return {'kp_driving': kp_video, 'kp_source': kp_appearance}

def train_data(configs):
  root_folder = configs['paths']['input']
  img_size = configs['hypers']["common"]['size']
  ext = configs['params']['ext']
  b_size = configs['hypers']["common"]['batch_size']


  dataset = DataClass(root_folder, img_size, ext)
  dataloader = DataLoader(dataset, batch_size = b_size, shuffle = False)

  in_features = configs["hypers"]["common"]["in_channel"]
  out_features = configs["hypers"]["common"]["out_channel"]
  max_features = configs["hypers"]["common"]["max_features"]
  num_block = configs["hypers"]["common"]["num_blocks"]
  block_expansion = configs["hypers"]["common"]["block_expansion"]
  kp_variance = configs["hypers"]["common"]["kp_variance"]
  mask_embedding_params = configs["hypers"]["generator_params"]["mask_embedding_params"]
  norm_const = configs["hypers"]["generator_params"]["norm_const"]


  kp_detector = KeyPointDetector(in_features, out_features, max_features, num_block, block_expansion)
  generator = Generator(in_features, out_features, max_features, num_block, block_expansion,\
    kp_variance, norm_const, **mask_embedding_params)

  for b, img in enumerate(dataloader, 0):
    img_size = img.size()
    img = img.view(-1, img_size[2], img_size[3], img_size[4], img_size[5]).permute(0, 2, 1, 3, 4)
    source_img = img[:, :, 0].unsqueeze(2)
    driving_img = img[:, :, 1].unsqueeze(2)
    kp = kp_detector(img)
    kp_split = split_kp(kp)
    predicted = generator(source_img, kp_split["kp_source"], kp_split["kp_driving"])
    e()
    # output = generate_kp(configs, img)

