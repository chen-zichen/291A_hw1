from urllib.response import addbase
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

### Do not modif the following codes
class ctx_noparamgrad(object):
    def __init__(self, module):
        self.prev_grad_state = get_param_grad_state(module)
        self.module = module
        set_param_grad_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_param_grad_state(self.module, self.prev_grad_state)
        return False
        
def get_param_grad_state(module):
    return {param: param.requires_grad for param in module.parameters()}

def set_param_grad_off(module):
    for param in module.parameters():
        param.requires_grad = False

def set_param_grad_state(module, grad_state):
    for param in module.parameters():
        param.requires_grad = grad_state[param]

### Ends


### PGD Attack
class PGDAttack():
    def __init__(self, step = 10, eps = 8 / 255, alpha = 0.01, loss_type = 'ce', targeted = True, 
                    num_classes = 10, norm = 'linf', fgsm = False):
        '''
        norm: this parameter means which type of l-p norm constraints we use for attack. Note that we onlyuse L-inf norm for our homework. 
        Therefore, this parameter is always linf.
        But if you are interested in implementing an l-2 norm bounded attack, you can also try to implement it. Note that in that case,
        the eps should be set to a larger value such as 200/255 because of the difference between l-2 and l-inf.
        '''
        self.attack_step = step
        self.eps = eps
        self.alpha = alpha
        self.loss_type = loss_type
        self.targeted = targeted
        self.num_classes = num_classes
        self.norm = norm
        self.fgsm = fgsm

    def ce_loss(self, logits, ys):
        loss = F.cross_entropy(logits, ys)
        if not self.targeted:
            loss = -loss
        return loss

    def cw_loss(self, logits, ys, x_adv, Xs):
        # loss = 0.* F.mse_loss(x_adv, Xs) + self.cw_margin(logits, ys, targeted=self.targeted)
        loss = self.cw_margin(logits, ys, targeted=self.targeted)
        return loss

    def cw_margin(self, logits, y, tau=0., targeted=False):
        one_hot_labels = torch.eye(len(logits[0]))[y].to(y)
        second, _ = torch.max((1-one_hot_labels)*logits, dim=1)
        first = torch.masked_select(logits, one_hot_labels.bool())

        if targeted:
            return torch.relu(second - first + tau).mean()
        else: 
            return torch.relu(first - second + tau).mean()

    def linf_proj(self, x, adv_x):
        adv_x = x + torch.clamp(adv_x - x, min=-self.eps, max=self.eps)
        raise adv_x

    def perturb(self, model, Xs, ys):

        delta = torch.empty_like(Xs).uniform_(-self.eps, self.eps)
        x_adv = torch.clamp(Xs + delta, min=0, max=1).detach()

        for _ in range(self.attack_step):
            x_adv.requires_grad = True
            model.zero_grad()
            logits = model(x_adv)

            # Calculate loss
            if self.loss_type == 'ce':
                loss = self.ce_loss(logits, ys)
            elif self.loss_type == 'cw':
                loss = self.cw_loss(logits, ys, x_adv, Xs)
            else: 
                raise NotImplementedError
            loss.backward()

            x_adv = x_adv.detach() - self.alpha*x_adv.grad.sign()
            
            delta = torch.clamp(x_adv - Xs, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(Xs + delta, min=0, max=1).detach()

        return x_adv - Xs



### FGSMAttack
'''
Theoretically you can transform your PGDAttack to FGSM Attack by controling some of its parameters like `attack_step`. 
If you do that, you do not need to implement FGSM in this class.
'''
class FGSMAttack():
    def __init__(self, eps = 8 / 255, loss_type = 'ce', targeted = True, num_classes = 10, norm = 'linf'):
        self.eps = eps
        self.loss_type = loss_type
        self.targeted = targeted
        self.num_classes = num_classes
        self.norm = norm

    def perturb(self, model: nn.Module, Xs, ys):
        Xs = Xs.clone()
        Xs.requires_grad=True
        model.zero_grad()

        logits = model(Xs)

        loss = F.cross_entropy(logits, ys)
        if self.targeted:
            loss = -loss

        loss.backward()
        x_adv = Xs.detach() + self.eps * Xs.grad.sign().detach()    
        x_adv = torch.clamp(x_adv, min=0, max=1).detach()

        return x_adv - Xs 

