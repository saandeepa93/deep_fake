import torch
import torch.nn.functional as F
import numpy as np
import cv2
from sys import exit as e


import utils as utils

# input = torch.arange(4*4).view(1, 1, 4, 4).float()

# Create grid to upsample input
# d = torch.linspace(-1, 1, 4)
# meshx, meshy = torch.meshgrid((d, d))
# grid = torch.stack((meshy, meshx), 2)
# grid = grid.unsqueeze(0) # add batch dim

# output = torch.nn.functional.grid_sample(input, grid, align_corners=True)
# print(input.size())
# print(grid.size())
# print(output.size())

# print(f"input------------{input.size()}\n, {input}")
# print("\n")
# print(f"grid------------{grid.size()}\n, {grid}")
# print("\n")
# print(f"output------------{output.size()}\n, {output}")
# print("\n")











img = cv2.resize(cv2.imread('./artifacts/0035_2m_-15P_10V_10H.jpg'), (200, 200))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
img = torch.from_numpy(img).permute(0, 1).unsqueeze(0).unsqueeze(1)

d = torch.linspace(-1, 1, 400)
d2 = torch.linspace(-1, 1, 500)
meshx, meshy = torch.meshgrid((d, d2))
grid = torch.stack((meshy, meshx), 2)
grid = grid.unsqueeze(0)
print(img.size(), grid.size())
int_img = F.grid_sample(img, grid, align_corners=True)



# bs, d, h_old, w_old, _ = grid.shape
# _, _, _, h, w = img.shape
# grid = grid.permute(0, 4, 1, 2, 3)
# grid = F.interpolate(grid, size=(d, h, w), mode = 'nearest')
# grid = grid.permute(0, 2, 3, 4, 1)



print("output", int_img.size())
utils.imshow(img.squeeze().numpy())
# utils.imshow(grid.numpy())
utils.imshow(int_img.squeeze().numpy())
e()

