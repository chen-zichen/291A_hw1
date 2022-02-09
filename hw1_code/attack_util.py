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


    def ce_loss(self, logits, ys, reduction = 'none'):
        ### Your code here

        ### Your code ends
        raise NotImplementedError

    def cw_loss(self, logits, ys, reduction = 'none'):
        ### Your code here

        ### Your code ends
        raise NotImplementedError

    def clamp(self, delta, lower, upper):
        ### Your code here

        ### Your code ends
        raise NotImplementedError

    def linf_proj(self, delta):
        ### Your code here

        ### Your code ends
        raise NotImplementedError

    def perturb(self, model: nn.Module, Xs, ys):
        delta = torch.ones_like(Xs)
        for iter_idx in range(self.attack_step):
            ### Your code here

            ### Your code ends
            pass
        return delta



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
        delta = torch.ones_like(Xs)
        ### Your code here

        ### Your code ends
        pass
        return delta        

