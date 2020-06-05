import cv2
import json



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



def reshape_tensor(img, gflag):
  img = img.unsqueeze(0)
  if gflag == 1:
    b, h, w = img.size()
    img = img.view(b, 1, h, w)
  else:
    b, h, w, c = img.size()
    img = img.view(b, c, h, w)
  return img


def reshape_array(img, gflag):
  img = img.numpy().squeeze()
  if gflag == 1:
    h, w = img.shape
    img = img.reshape(h, w, 1)
  else:
    c, h, w = img.shape
    img = img.reshape(h, w, c)
  return img