# 1. test standard model and robust model
python3 evaluate.py --model_name resnet_cifar10.pth --noattack
python3 evaluate.py --model_name pgd10_eps8.pth --noattack

# standard model version
# 2. fgsm + untargeted ce
python3 evaluate.py --model_name resnet_cifar10.pth --fgsm

# 3. pgd + untargeted ce + t=10
python3 evaluate.py --model_name resnet_cifar10.pth --attack_step 10

# 4. pgd + untargeted cw + t=10
python3 evaluate.py --model_name resnet_cifar10.pth --attack_step 10 --loss_type cw

# 5. pgd + targeted cw + t=10
python3 evaluate.py --model_name resnet_cifar10.pth --attack_step 1 --loss_type cw --targeted

# 6. pgd + ce + different settings... example
python3 evaluate.py --model_name resnet_cifar10.pth --attack_step 20 --loss_type ce 
python3 evaluate.py --model_name resnet_cifar10.pth --attack_step 10 --loss_type ce --eps 1
