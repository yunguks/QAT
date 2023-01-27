from models import quat_mobilenet_v2, MobileNet_V2_Weights, quantize_model
from models import Quant_ReLU,Quant_ReLU6
from torchsummary import summary
from utils import Data
from utils.Train import Training, Evaluating
import torch
import argparse
import os
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="QAT training test")
    parser.add_argument("--model",type=str,default="mobilenet", help="select model. Default mobilenet")
    parser.add_argument("--lr", type=float,default=1e-6, help="initial lr. Default 1e-6")
    parser.add_argument("--random", type=int,default=42, help="select random seed. Default 42") 
    parser.add_argument("--weights", type=str, help="load weights file name")
    parser.add_argument("--tiny", action='store_true', help="select tiny model.")
    parser.add_argument("--act", action='store_true', help="using layer.Quant_ReLU6 with activation output quant")
    parser.add_argument("--pretrain", type=bool,default=True, help="using cifar10 pretrain weights. Default True")
    parser.add_argument("--savename", type=str, default=None, help="Name to store the model. Default test.pt")
    return vars(parser.parse_args())

if __name__=="__main__":
    kargs = get_args()
    print(kargs)
    file_name = kargs["model"]+"_"+str(kargs["random"])+".csv"
  
    # device 
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    
    if kargs["act"]:
        activation_layer = Quant_ReLU6
    else:
        activation_layer = torch.nn.ReLU6
    
    # model load
    if kargs["tiny"]:
        NEW_MODEL = quat_mobilenet_v2(cifar10=True)
        if kargs["pretrain"]:
            NEW_MODEL.load_state_dict(torch.load("./models/weights/tiny_mobilenetv2_cifar.pt"))
    else:
        NEW_MODEL = quat_mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1,activation_layer=activation_layer)
        NEW_MODEL.classifier.append(torch.nn.Dropout(0.2))
        NEW_MODEL.classifier.append(torch.nn.Linear(1000, 10))
        if kargs["pretrain"]:
            NEW_MODEL.load_state_dict(torch.load("./models/weights/q_mobilenetv2_cifar10.pt"))
        
    if kargs["weights"]:
        NEW_MODEL.load_state_dict(torch.load(kargs["weights"]))
        
    # data load
    train_loader, test_loader = Data.Cifar10_Dataloader()
    
    # optimizer 
    optimizer = torch.optim.SGD(NEW_MODEL.parameters(),lr=kargs["lr"],momentum=0.9)
    
    # # scheduler 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,60,90], gamma=0.5)
     
    # # train model
    NEW_MODEL = Training(NEW_MODEL,train_loader,test_loader,gpu_device,optimizer,scheduler,save_name=kargs["savename"])
