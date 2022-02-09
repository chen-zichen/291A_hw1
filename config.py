import argparse
from ast import arg
from distutils.command import config
import json
import utils

def parse_args():
    describe = 'assignment1 - evaluate the adversarial robustness'
    parser = argparse.ArgumentParser(description=describe)
    parser.add_argument('--config', type=str, default='config.json', help='config file path')

    return parser.parse_args()

def get_config(config_path):
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    arg_dict = vars(arg)
    for key in arg_dict : 
        if key in configs:
            if arg_dict[key] is not None:
                configs[key] = arg_dict[key]
    configs = utils.ConfigMapper(configs)
    return configs

    
