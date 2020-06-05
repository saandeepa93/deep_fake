from sys import exit as e
import numpy as np

import modules.utils as utils
from modules.keypoint.kp_main import generate_kp
from modules.preprocessor import read_data

def main():
  in_channel = utils.get_config("in_channel")
  out_channel = utils.get_config("out_channel")
  input_path = utils.get_config("input_path")
  # read_data(input_path, in_channel)
  generate_kp(in_channel, out_channel)

if __name__ == '__main__':
  main()