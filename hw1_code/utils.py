# import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
# from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
import numpy as np

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 2
    # train_dataset = datasets.CIFAR10(
        # dir_, train=True, transform=train_transform, download=True)
    
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
        
    # prepare eval data
    val_ratio = 0.1
    train_size = int(50000 * (1 - val_ratio))
    val_size = 50000 - train_size

    train_dataset = Subset(CIFAR10(dir_, train=True, transform=train_transform, download=True), 
                     list(range(train_size)))
    val_set = Subset(CIFAR10(dir_, train=True, transform=test_transform, download=True),
                     list(range(train_size, train_size + val_size)))
    

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    return train_loader, val_loader, test_loader, dataset_normalization

class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)
        
def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            # if opt is not None:
            #     with amp.scale_loss(loss, opt) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     loss.backward()
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            # ce
            # loss = ce_loss(output, y)
            # cw
            loss = cw_loss(output, y)

            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

def ce_loss(logits: torch.float, ys: torch.int):
    # logits: outputs of model(adv_x), ys: label of xs 
    # ys shape: bs
    # logits shape: bs x 10
    loss = F.cross_entropy(logits, ys)
    loss = -loss
    return loss

def cw_loss(logits: torch.float, ys: torch.int):
    # logits: outputs of model(x), ys: label of xs 
    # ys shape: bs 
    # logits shape: bs x 10
    loss = cw_margin(logits, ys, targeted=False)
    return loss

def cw_margin(logits: torch.float, ys: torch.int, tau=0., targeted=False):
    # logits: outputs of model(x), ys: label of xs
    # one on to the diagonal,rest is 0: c x c

    # one hot label
    # the victime label will be assigned class [1], others are 0
    one_hot_labels = F.one_hot(ys, num_classes=10).to(ys)

    # get each logits'wrong score
    wrong_logit, _ = torch.max((1-one_hot_labels)*logits, dim=1)

    # only get targeted confidence score
    correct_logit = torch.masked_select(logits, one_hot_labels.bool())
    # if target, larger the difference of correct, narrow to the target label
    if targeted:
        # close to the target label, max perturbed input and true label
        # loss = -(correct - wrong)
        return torch.relu(wrong_logit - correct_logit + tau).mean()
    else: 
        # loss = correct - wrong
        # away from individual true label
        return torch.relu(correct_logit - wrong_logit + tau).mean()
