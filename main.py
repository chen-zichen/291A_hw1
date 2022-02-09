import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


from config import *
from attack import *
from src.eval import Evaluator
from hw1_code import model_util

def main():
    args = parse_args()
    configs = get_config('config/ut_fgsm_ut_ce.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model= model_util.ResNet18()
    model.load_state_dict(torch.load(configs.model_name, map_location=device), strict=False)
    # pretrained = torch.load_state_dict(configs.model_name, map_location=device)
    # torch.load(configs.model_name, map_location=device)
    evaluation = Evaluator(configs, model)
    evaluation.eval_model()

if __name__ == '__main__':
    main()



