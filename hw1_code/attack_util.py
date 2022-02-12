from urllib.response import addbase
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

### Do not modif the following codes
class ctx_noparamgrad(object):
    def __init__(self, module):
        # get_param_grad_state for set_param_grad_state func, need prev state
        self.prev_grad_state = get_param_grad_state(module)
        self.module = module
        # set parm in module no grad
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

    def ce_loss(self, logits: torch.float, ys: torch.int):
        # logits: outputs of model(x), ys: label of xs 
        # ys shape: bs x 1
        # logits shape: bs x 10
        loss = F.cross_entropy(logits, ys)
        if not self.targeted:
            loss = -loss
        return loss

    def cw_loss(self, logits: torch.float, ys: torch.int):
        # logits: outputs of model(x), ys: label of xs 
        # ys shape: bs x 1
        # logits shape: bs x 10
        loss = self.cw_margin(logits, ys, targeted=self.targeted)
        return loss

    def cw_margin(self, logits: torch.float, ys: torch.int, tau=0., targeted=False):
        # logits: outputs of model(x), ys: label of xs
        # one on to the diagonal,rest is 0: c x c

        # one hot label
        # the victime label will be assigned class [1], others are 0
        one_hot_labels = F.one_hot(ys, num_classes=10).to(ys)

        # get each logits'wrong score
        wrong_logit, _ = torch.max((1-one_hot_labels)*logits, dim=1)

        # only get targeted confidence score
        correct_logit = torch.masked_select(logits, one_hot_labels.bool())

        if targeted:
            # loss = -(correct - wrong)
            return torch.relu(wrong_logit - correct_logit + tau).mean()
        else: 
            # loss = correct - wrong
            return torch.relu(correct_logit - wrong_logit + tau).mean()


    def perturb(self, model:nn.Module, Xs: torch.float, ys: torch.int):
        """
        model: resnet18
        Xs: data - image input, shape: bs 3 32 32  
        ys: attack_labels. (targeted=1)
        """
        # delta: pertubation, shape: bs 3 32 32, range [-eps,eps]
        delta = torch.empty_like(Xs).uniform_(-self.eps, self.eps)

        # x_adv: adverserial input, norm
        x_adv = torch.clamp(Xs + delta, min=0, max=1).detach()

        for _ in range(self.attack_step):
            # set x_adv grad
            x_adv.requires_grad = True
            # eval model
            model.zero_grad()
            # logits is the output of model with adverserial input
            # logits shape: bs x c [64,10]
            logits = model(x_adv)
            

            # Calculate loss
            if self.loss_type == 'ce':
                loss = self.ce_loss(logits, ys)
            elif self.loss_type == 'cw':
                loss = self.cw_loss(logits, ys)
            else: 
                raise NotImplementedError
            loss.backward()

            # update adverserial input with adverserial input grad sign
            x_adv = x_adv.detach() - self.alpha*x_adv.grad.sign()
            
            # update pertubation delta, min(delta), with norm
            delta = torch.clamp(x_adv - Xs, min=-self.eps, max=self.eps)
            # update adverserial input, with norm
            x_adv = torch.clamp(Xs + delta, min=0, max=1).detach()
        # return pertubation
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

    def perturb(self, model: nn.Module, Xs: torch.float, ys: torch.int):
        """
        model: resnet18
        Xs: data - image input, shape: bs 3 32 32  
        ys: attack_labels. (targeted=1)
        """

        Xs = Xs.clone()
        # set grad to orignal data/input
        Xs.requires_grad=True
        # set model eval
        model.zero_grad()

        # logits is the output of model with adverserial input
        # logits shape: bs x c [64,10]
        logits = model(Xs)

        # ce loss, compare confidence score with attack label 
        loss = F.cross_entropy(logits, ys)
        if self.targeted:
            loss = -loss

        loss.backward()

        # update adverserial input with original data and grad
        x_adv = Xs.detach() + self.eps * Xs.grad.sign().detach()  
        # norm  
        x_adv = torch.clamp(x_adv, min=0, max=1).detach()
        # return pertubation 
        return x_adv - Xs 

