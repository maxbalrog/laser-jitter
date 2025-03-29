'''
Utility functions
'''
import torch
import numpy as np
import random
import yaml


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def read_yaml(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)
            return exc


def write_yaml(yaml_file, data):
    with open(yaml_file, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)