import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preact_resnet import PreActResNet18, ResNet18

from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard)

def test(model, test_loader):

    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 1)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    print('pgd acc: ', pgd_acc)
    print('clean acc: ', test_acc)
    # logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    # logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == '__main__':
    model_path = 'train_pgd_output/model-fast-new.pth'
    # data
    data_dir = 'data' 
    _, _, test_loader, _ = get_loaders(data_dir, 64)
    model = PreActResNet18().cuda()
    # model = ResNet18(num_classes=10).cuda()
    # model.normalize = norm
    model.load_state_dict(torch.load(model_path))
    
    model = model.cuda()

    test(model, test_loader)