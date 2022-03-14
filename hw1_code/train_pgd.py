import argparse
import logging
import os
import time

# import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preact_resnet import PreActResNet18, ResNet18
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--data-dir', default='data', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=1, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=2, type=int)
    parser.add_argument('--alpha', default=2, type=int, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_pgd_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output-AT-1.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, val_loader, test_loader, norm_layer = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    model = PreActResNet18().cuda()
    # model = ResNet18(num_classes=10).cuda()
    # model.normalize = norm_layer
    model.train()

    # opt
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc \t Robust')
    for epoch in range(args.epochs):

        for phase in ['train', 'val']: 
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloaders = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloaders = val_loader

            start_epoch_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0

            for i, (X, y) in enumerate(dataloaders):
                X, y = X.cuda(), y.cuda()
                delta = torch.zeros_like(X).cuda()
                

                if args.delta_init == 'random':
                    for i in range(len(epsilon)):
                        delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.requires_grad = True
                model.eval()
                for _ in range(args.attack_iters):
                    
                    output = model(X + delta)
                    loss = criterion(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta.grad.zero_()
                    delta.requires_grad = True
                delta = delta.detach()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(X + delta)
                    loss = criterion(output, y)
                    if phase == 'train':
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        scheduler.step()

                    train_loss += loss.item() * y.size(0)
                    train_acc += (output.max(1)[1] == y).sum().item()
                    train_n += y.size(0)
                    # scheduler.step()
            epoch_time = time.time()
            lr = scheduler.get_lr()[0]
            pgd_loss, pgd_acc = evaluate_pgd(val_loader, model, 1, 1)

            logger.info('%s \t %d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f',
                phase, epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n, pgd_acc)


        if epoch ==19:
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'model-at-20.pth'))

    train_time = time.time()
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model-at.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 1)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == "__main__":
    main()
