from sys import exit as e
import numpy as np
import yaml
from sys import exit as e
import click

import modules.util as util
# from modules.preprocessor import read_data
from modules.train import train_data


@click.command()
@click.option('--config', help = "path of config file")
def train(config):
  # read_data(configs)
  configs = util.get_config(config)
  train_data(configs)


@click.group()
def main():
  pass


if __name__ == '__main__':
  main.add_command(train)
  main()