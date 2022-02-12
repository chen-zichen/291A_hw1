from cProfile import label
import torch
import data_util
import model_util
import attack_util
from attack_util import ctx_noparamgrad

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--eps", type = float, default = 8.0)
parser.add_argument("--alpha", type = float, default = 2.0)
parser.add_argument("--attack_rs", type = int, default = 1)
parser.add_argument("--attack_step", type = int, default = 10)
parser.add_argument("--loss_type", type = str, default = "ce", choices = ['ce','cw'])
parser.add_argument('--data_dir', default='./data/', type=str, help="The folder where you store your dataset")
parser.add_argument('--model_prefix', default='./checkpoints/',
                    help='File folders where you want to store your checkpoints')
parser.add_argument('--model_name', default='resnet_cifar10.pth',
                    help='File folders where you want to store your checkpoints')
parser.add_argument("--fgsm", action = 'store_true')
parser.add_argument("--targeted", action = 'store_true')
args = parser.parse_args()

device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
train_loader, valid_loader, test_loader, norm_layer = data_util.cifar10_dataloader(data_dir = args.data_dir)
num_classes = 10
model = model_util.ResNet18(num_classes = num_classes)
model.normalize = norm_layer
model_path = args.model_prefix + args.model_name
model.load(model_path)

model = model.to(device)

attack_step = args.attack_step
eps = args.eps / 255.
restart = args.attack_rs
alpha = args.alpha / 255.
fgsm = args.fgsm
loss_type = args.loss_type
targeted = args.targeted

### Your code here for creating the attacker object
if args.fgsm:
    attacker = attack_util.FGSMAttack(eps = eps, loss_type = loss_type, targeted = targeted, num_classes = num_classes, norm = 'linf')
else:
    attacker = attack_util.PGDAttack(step = attack_step, eps = eps, alpha = alpha, 
                                    loss_type = loss_type, targeted = targeted, num_classes = num_classes, 
                                    norm = 'linf', fgsm = fgsm)

### Your code ends

total = 0
clean_correct_num = 0
robust_correct_num = 0
target_label = 1 ## only for targeted attack


## Make sure the model is in `eval` mode. Otherwise some operations such as dropout will  
model.eval()

for data, labels in tqdm(test_loader):
    data = data.float().to(device)
    labels = labels.to(device)
    if targeted:
        # set false to these (data label ≠ target label) 
        data_mask = (labels != target_label)
        # data shape: 61 3 32 32 
        # masked data/labels
        data = data[data_mask]
        labels = labels[data_mask]
        # copy label shape 
        attack_labels = torch.ones_like(labels).to(device)
    else:
        attack_labels = labels
    attack_labels = attack_labels.to(device)
    # number of data
    batch_size = data.size(0)

    # for final calculation 
    total += batch_size
    with ctx_noparamgrad(model):
        # no grad
        # generate perturbation
        # perturbed_data + original data (image)
        perturbed_data = attacker.perturb(model, data, attack_labels) + data

        # clean model acc
        predictions = model(data)
        # predictions shape bs x c
        # argmax keep the bs x 1 to match the 'real labels'
        clean_correct_num += torch.sum(torch.argmax(predictions, dim = -1) == labels).item()

        # robust acc 
        predictions = model(perturbed_data)
        robust_correct_num += torch.sum(torch.argmax(predictions, dim = -1) == labels).item()

print(f"total {total}, correct {clean_correct_num}, adversarial correct {robust_correct_num}, clean accuracy {clean_correct_num / total}, robust accuracy {robust_correct_num / total}")
