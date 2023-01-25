from models import quat_mobilenet_v2, MobileNet_V2_Weights, quantize_model
from models import Quant_ReLU
from torchsummary import summary
from utils import Data
from utils.Train import Training, Evaluating
import torch
import argparse
import os
import pandas as pd
import copy

def get_args():
    parser = argparse.ArgumentParser(description="QAT training test")
    parser.add_argument("--model",type=str,default="mobilenet", help="select model. Default mobilenet")
    parser.add_argument("--lr", type=float, default= 1e-2,help="initial learning rate. Default 1e-2")
    parser.add_argument("--random", type=int,default=42, help="select random seed. Default 42") 
    parser.add_argument("--tiny", type=bool, default=False, help="select tiny model. Default False")
    return vars(parser.parse_args())

if __name__=="__main__":
    kargs = get_args()
    print(kargs)
    file_name = kargs["model"]+"_"+str(kargs["random"])+".csv"
  
    # device 
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    
    # model load
    if kargs["tiny"]:
        NEW_MODEL = quat_mobilenet_v2(cifar10=True)
        NEW_MODEL.load_state_dict(torch.load("./models/tiny_mobilenetv2_cifar.pt"))
    else:
        NEW_MODEL = quat_mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1,activation_layer=torch.nn.ReLU)
        NEW_MODEL.classifier.append(torch.nn.Dropout(0.2))
        NEW_MODEL.classifier.append(torch.nn.Linear(1000, 10))
        NEW_MODEL.load_state_dict(torch.load("./models/mobilenetv2_cifar10.pt"))
    # data load
    train_loader, test_loader = Data.Cifar10_Dataloader()
    
    # optimizer 
    # optimizer = torch.optim.SGD(NEW_MODEL.parameters(),lr=1e-5,momentum=0.9)
    
    # # scheduler 
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,60,90], gamma=0.5)
     
    # # train model
    # NEW_MODEL = Train.Training(NEW_MODEL,train_dataloader,test_loader,gpu_device,optimizer,scheduler)
    NEW_MODEL.to(cpu_device)
    NEW_MODEL.eval()
    _, val_acc = Evaluating(NEW_MODEL, test_loader,device=cpu_device)
    print(f"Before Model acc : {val_acc:.2f}%")
    
    order = {}
    if kargs["tiny"]:
        order["Post"] = "tiny_mobilenetv2_cifar.pt"
    else:
        order["Post"] = "mobilenetv2_cifar10.pt"
        order["QAT"] = "QAT_mobilenetv2_cifar10.pt"
        order["Input QAT"] = "input_q_mobilenetv2_cifar10.pt"
        order["Weight QAT"] = "weight_q_mobilenetv2_cifar10.pt"
        order["Input-weight QAT"] = "input_weight_q_mobilenetv2_cifar10.pt"
    qconfig = []
    qconfig.append(torch.quantization.get_default_qconfig("fbgemm"))
    n = torch.ao.quantization.QConfig(  # type: ignore[assignment]
                activation=torch.ao.quantization.default_observer,
                weight=torch.ao.quantization.default_per_channel_weight_observer
            )
    qconfig.append(n)
    n =torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.default_observer, 
            weight=torch.ao.quantization.default_weight_observer
            )
    qconfig.append(n)
    n =torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.default_observer, 
            weight=torch.ao.quantization.weight_observer_range_neg_127_to_127
            )
    qconfig.append(n)
    
    with torch.no_grad():
        for key in order.keys():
            new_model = copy.deepcopy(NEW_MODEL)
            print(f"----------------- {key} quantization -----------------")
            model_name = "./models/"+order[key]
            if "QAT" == key:
                jit_model = torch.jit.load("./models/Q_mobilenetv2_cifar10_jit.pt")
                _,int8_acc = Evaluating(jit_model, test_loader,"cpu")
                q = torch.quantization.get_default_qconfig("fbgemm")
                print(f"{q} : {int8_acc:.2f}acc",end="\n\n")
            else:
                new_model.load_state_dict(torch.load(model_name))
                for q in qconfig:
                    default_q = copy.deepcopy(new_model)
                    quantize_model(default_q, data= train_loader,qconfig=q)
                    default_q = torch.jit.script(default_q)
                    _,int8_acc = Evaluating(default_q,test_loader,"cpu")
                    print(f"--> {int8_acc:.2f}acc",end="\n\n")
        