import os
import torch
import torch.nn.functional as F
from attack import Attacker

class FastGradientSignAttack(Attacker.attacker):
    def __init__(self, model, config):
        super(FastGradientSignAttack, self).__init__(model, config)
        # self.target = target

    def forward(self, x, y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        x_adv = x.detach().clone()

        x_adv.requires_grad=True
        self.model.zero_grad()

        out = self.model(x_adv)

        if self.loss_type == "ce":
            loss = F.cross_entropy(out, y)
        # todo loss + cw / untargeted cw

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        loss.backward()
        x_adv = x_adv.detach() - self.eps * x_adv.grad.sign()        
        x_adv = torch.clamp(x_adv, *self.clamp)

        return x_adv

