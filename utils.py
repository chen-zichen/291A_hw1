import torch
import torch.nn as nn

class ConfigMapper(object):
    def __init__(self, args):
        for key in args:
            self.__dict__[key] = args[key]

class CWloss():
    def __init__(self, target=False):
        super(CWloss, self).__init__(target)
        self.target = target

    def cw_loss(self, labels=None, logits=None):

        # Carlini-Wagner loss
        loss_out = self.margin(logits, labels, targeted=self.target)
        return nn.ReLU(loss_out)
    
    def margin(self, logits, y, delta, targeted=False):
        """
        :param logits: A tensor of shape [N, C] representing the
        logits output from the classifier.
        :param y: A tensor of shape [N] representing the target
        labels.
        :param delta: A scalar representing the desired margin.
        :param targeted: A boolean indicating whether to compute
        the targeted or untargeted loss.
        :return: A tensor of shape [N] representing the loss.
        """
        if targeted:
            return torch.sum(torch.max(torch.zeros(logits.size()).cuda(), 
                logits[range(logits.size(0)), y] - logits + delta))
        else:
            return torch.sum(torch.max(torch.zeros(logits.size()).cuda(), 
                logits - logits[range(logits.size(0)), y] + delta))
