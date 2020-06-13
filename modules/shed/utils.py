from torch import nn
import torch
import torch.nn.functional as F
import cv2


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


def get_config(keys):
  f = open('config.json')
  config_data = json.load(f)
  return config_data[keys]





class UpBlock3D(nn.Module):
    """
    Simple block for processing video (decoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock3D, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = nn.BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        print("x shape", x.size())
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        print("out shape", out.size())
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out




class DownBlock3D(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DownBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = nn.BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out
