import torch
from torch.autograd import Variable
from torch.nn import Conv3d
import matplotlib.pyplot as plt

def show(img):
  plt.imshow(img)
  plt.show()

inp = torch.randn(32, 3, 1, 64, 64)
inp = Variable(inp)

conv = Conv3d(3, 3, (1, 3, 3), groups = 3, bias = False)

out = conv(inp)
print(out.size())

print(conv.weight.size())
