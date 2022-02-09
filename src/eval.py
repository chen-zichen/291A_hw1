from distutils.command import config
import torch
import torchvision.transforms as T
import torchvision
import os
from utils import CWloss

# from attack import *
import attack
from attack import fgsm
# from attack import FastGradientSignAttack
# from attack import ProjectedGradientDescentAttack
class Evaluator:
    def __init__(self, configs, model):
        self.configs = configs
        self.model = model
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])
        self.model.to(self.device)

        # load data
        self.test_dataset = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=configs.data_dir, train=False, download=True, 
            transform=transform_test), batch_size=64, shuffle=True, 
            num_workers=4
        )
        
        # set attack configs
        attack_configs = {
            "eps": configs.eps/255.0,
            "attack_steps": configs.attack_steps,
        }

        # attack method
        if configs.attack == "fgsm":
            self.attacker = fgsm.FastGradientSignAttack(self.model, attack_configs)
        elif configs.attack == "pgd":
            self.attacker = attack.fgsm.ProjectedGradientDescentAttack(self.model, attack_configs)   # todo
        else:
            raise ValueError("Attack type not supported")
        # loss function
        if configs.loss_type == "ce":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif configs.loss_type == "cw": 
            self.loss_fn = CWloss()   # todo: add cw

    def eval_model(self):
        self.model.eval()
        correct = 0
        adv_correct = 0
        for i, (image,label) in enumerate(self.test_dataset):
            image = image.to(self.device)
            label = label.to(self.device)

            adv_image = self.attacker(image, label)
            adv_image = adv_image.to(self.device)

            output = self.model(adv_image)

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum()
            adv_output = self.model(image)
            _, adv_predicted = torch.max(adv_output.data, 1)
            adv_correct += (adv_predicted == label).sum()

        print("Accuracy of the network on the test images: %d %% " % (100 * correct / len(self.test_dataset)))
        print("Accuracy of the network on the adversarial images: %d %% " % (100 * adv_correct / len(self.test_dataset)))
        return correct/len(self.test_dataset), adv_correct/len(self.test_dataset)