import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import glob
import os
from natsort import natsorted
from sys import exit as e

import modules.util as util

class DataClass(Dataset):
  def __init__(self, root_folder, isize, ext, set_size=8):
    super(DataClass, self).__init__()
    self.set_size = set_size
    self.root_folder = root_folder
    self.all_files = natsorted(glob.glob(os.path.join(self.root_folder, "*"+ext)))
    self.isize = isize
    self.ext = ext
    self.transform = transforms.Compose(
      [transforms.ToPILImage(),
      transforms.Resize((isize, isize)),
      transforms.ToTensor(),
      transforms.Normalize((0, 0, 0), (1, 1, 1))]
    )


  def __len__(self):
    return len(self.all_files) // self.set_size


  def __getitem__(self, idx):
    i = idx * self.set_size
    j = torch.LongTensor(1).random_(i, i + self.set_size-1)[0]
    final = None

    lst = torch.Tensor([k for k in range(i, i + self.set_size)]).type(torch.int)
    lst = lst[torch.randperm(self.set_size)]
    for i in range(0, (self.set_size), 2):
      file_name1 = self.all_files[lst[i]]
      file_name2 = self.all_files[lst[i+1]]

      img1 = io.imread(file_name1)
      img2 = io.imread(file_name2)

      img1 = self.transform(img1).unsqueeze(0)
      img2 = self.transform(img2).unsqueeze(0)

      temp = torch.cat((img1, img2), 0).unsqueeze(0)
      if final is None:
        final = temp
      else:
        final = torch.cat((final, temp))
    return final