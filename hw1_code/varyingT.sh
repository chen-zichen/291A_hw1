python3 evaluate.py --model_name resnet_cifar10.pth --attack_step 10 --loss_type ce --eps 1
python3 evaluate.py --model_name resnet_cifar10.pth --attack_step 10 --loss_type ce --eps 2
python3 evaluate.py --model_name resnet_cifar10.pth --attack_step 10 --loss_type ce --eps 4
python3 evaluate.py --model_name resnet_cifar10.pth --attack_step 10 --loss_type ce --eps 6
python3 evaluate.py --model_name resnet_cifar10.pth --attack_step 10 --loss_type ce --eps 8

python3 evaluate.py --model_name pgd10_eps8.pth --attack_step 10 --loss_type ce --eps 1
python3 evaluate.py --model_name pgd10_eps8.pth --attack_step 10 --loss_type ce --eps 2
python3 evaluate.py --model_name pgd10_eps8.pth --attack_step 10 --loss_type ce --eps 4
python3 evaluate.py --model_name pgd10_eps8.pth --attack_step 10 --loss_type ce --eps 6
python3 evaluate.py --model_name pgd10_eps8.pth --attack_step 10 --loss_type ce --eps 8