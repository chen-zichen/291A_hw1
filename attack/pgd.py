from attack import Attacker
import torch
import torch.nn.functional as F


class ProjectedGradientDescentAttack(Attacker.attacker):
    def __init__(self, model, config, target=None):
        super(ProjectedGradientDescentAttack, self).__init__(model, config)
        self.target = target

    def forward(self, x, y):
        # Projected Gradient Descent Attack
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target: Target label 
        :return adversarial image
        """
        x_adv = x.detach().clone()

        for i in range(self.config['attack_steps']):
            x_adv.requires_grad=True
            self.model.zero_grad()
            logits = self.model(x_adv)


            if self.config['loss_type'] == "ce":
                loss = F.cross_entropy(logits, y)
                loss.backward()
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv.detach() - self.config['alpha']/255 * grad

            if self.config['loss_type'] == "cw":
                # todo loss + cw / untargeted cw
                if self.target is None:
                    loss = F.cross_entropy(logits, y)
                else:
                    loss = F.cross_entropy(logits, self.target)
                loss.backward()
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv.detach() - self.config['alpha']/255 * grad

            # projection 
            x_adv = x + torch.clamp(x_adv - x, min=-self.config['eps']/255, max=self.config['eps']/255)
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *self.clamp)
        return x_adv
