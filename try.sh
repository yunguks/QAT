
# torch QAT
nohup python3 train.py --seed 10 --result --device 0 --torchQAT --savename "./result/random/TorchQAT_mobilenetv2_cifar10_10.pt" 1>/dev/null 2> err.txt & 
nohup python3 train.py --seed 20 --result --device 0 --torchQAT --savename "./result/random/TorchQAT_mobilenetv2_cifar10_20.pt" 1>/dev/null 2> err.txt & 
nohup python3 train.py --seed 30 --result --device 0 --torchQAT --savename "./result/random/TorchQAT_mobilenetv2_cifar10_30.pt" 1>/dev/null 2> err.txt & 

# origin model
# nohup python3 train.py --seed 10 --result --device 0 --savename "./result/random/mobilenetv2_cifar10_10.pt" 1>/dev/null 2> err.txt & 
# nohup python3 train.py --seed 20 --result --device 0 --savename "./result/random/mobilenetv2_cifar10_20.pt" 1>/dev/null 2> err.txt & 
# nohup python3 train.py --seed 30 --result --device 0 --savename "./result/random/mobilenetv2_cifar10_30.pt" 1>/dev/null 2> err.txt & 

# my qat
# nohup python3 train.py --seed 10 --input-fake --act --result --device 1 --savename "./result/random/actq_mobilenetv2_cifar10_10.pt" 1>/dev/null 2> err.txt & 
# nohup python3 train.py --seed 20 --input-fake --act --result --device 1 --savename "./result/random/actq_mobilenetv2_cifar10_20.pt" 1>/dev/null 2> err.txt & 
# nohup python3 train.py --seed 30 --input-fake --act --result --device 1 --savename "./result/random/actq_mobilenetv2_cifar10_30.pt" 1>/dev/null 2> err.txt & 

# nohup python3 train.py --seed 10 --weight-fake --result --device 1 --savename "./result/random/weightq_mobilenetv2_cifar10_10.pt" 1>/dev/null 2> err.txt & 
# nohup python3 train.py --seed 20 --weight-fake --result --device 1 --savename "./result/random/weightq_mobilenetv2_cifar10_20.pt" 1>/dev/null 2> err.txt & 
# nohup python3 train.py --seed 30 --weight-fake --result --device 1 --savename "./result/random/weightq_mobilenetv2_cifar10_30.pt" 1>/dev/null 2> err.txt & 

# nohup python3 train.py --seed 10 --fake --result --device 0 --savename "./result/random/fake_mobilenetv2_cifar10_10" 1>/dev/null 2> err.txt & 
# nohup python3 train.py --seed 20 --fake --result --device 0 --savename "./result/random/fake_mobilenetv2_cifar10_20" 1>/dev/null 2> err.txt & 
# nohup python3 train.py --seed 30 --fake --result --device 0 --savename "./result/random/fake_mobilenetv2_cifar10_30" 1>/dev/null 2> err.txt & 
