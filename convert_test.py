from models import quat_mobilenet_v2, MobileNet_V2_Weights, quantize_model
from models import Quant_ReLU
from utils import Data
from utils.Train import Evaluating
import torch
import argparse
import os
import pandas as pd
import copy

def get_args():
    parser = argparse.ArgumentParser(description="QAT training test")
    parser.add_argument("--model",type=str,default="mobilenet", help="select model. Default mobilenet")
    parser.add_argument("--random", type=int,default=42, help="select random seed. Default 42") 
    parser.add_argument("--weight", type=str, help="load weights file name")
    parser.add_argument("--tiny",action="store_true", help="select tiny model. Default False")
    parser.add_argument("--only",action="store_true", help="test only one post quant")
    parser.add_argument("--path",type=str, default=None, help="convert test file path")
    parser.add_argument("--savename", type=str, default=None, help="save scv file and the file name")
    parser.add_argument("--loop", type=int, default=1,help="try loop time convert")
    return vars(parser.parse_args())

if __name__=="__main__":
    kargs = get_args()
    print(kargs)
  
    # device 
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    
    # model load
    if kargs["tiny"]:
        NEW_MODEL = quat_mobilenet_v2(cifar10=True)
    else:
        NEW_MODEL = quat_mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1,activation_layer=torch.nn.ReLU)
        NEW_MODEL.classifier.append(torch.nn.Dropout(0.2))
        NEW_MODEL.classifier.append(torch.nn.Linear(1000, 10))
    # data load
    if kargs["weight"]:
        NEW_MODEL.load_state_dict(torch.load(kargs["weight"]))
        
    train_loader, test_loader = Data.Cifar10_Dataloader()
    
    NEW_MODEL.to(cpu_device)
    NEW_MODEL.eval()
    if kargs["savename"]:
        if kargs["path"]:
            csv_name = os.path.join(kargs["path"],kargs["savename"])
        else:
            csv_name = kargs['savename']
        header = pd.DataFrame(columns=["filename",
                                       "origin_acc",
                                       "act=hist,weight=channelMinMax",
                                       "act=hist,weight=MinMax",
                                       "act=MinMax,weight=channelMinMax",
                                       "act=MinMax,weight=MinMax"])
        if os.path.isfile(csv_name) is False:
            header.to_csv(csv_name,index=False)
            
    order = {}
    if kargs["tiny"]:
        if kargs["weight"] is None:
            order["Post"] = ["./models/weights/tiny_mobilenetv2_cifar.pt"]
        else:
            order["Post"] = [kargs["weight"]]
        
    elif kargs["only"]:
        order["Post"] = [kargs["weight"]]
    
    elif kargs["path"]:
        filelist = os.listdir(kargs["path"])
        filelist_pt = [f for f in filelist if f.endswith(".pt")]
        order["Post"] = [os.path.join(kargs["path"],name) for name in filelist_pt]
    else:
        order["Post"] = ["./models/weights/q_mobilenetv2_cifar10.pt"]
        order["QAT"] = ["./models/weights/QAT_mobilenetv2_cifar10.pt"]
        order["Input QAT"] = ["./models/weights/input_q_mobilenetv2_cifar10.pt"]
        order["Weight QAT"] = ["./models/weights/weight_q_mobilenetv2_cifar10.pt"]
        order["Input-weight QAT"] = ["./models/weights/input_weight_q_mobilenetv2_cifar10.pt"]
       
    
    qconfig = []
    qconfig.append(torch.quantization.get_default_qconfig("fbgemm"))
    n = torch.ao.quantization.QConfig(  # type: ignore[assignment]
                activation=torch.ao.quantization.default_histogram_observer,
                weight=torch.ao.quantization.default_weight_observer
            )
    qconfig.append(n)
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
    
    with torch.no_grad():
        for key in order.keys():
            print(f"----------------- {key} quantization -----------------")
            for file in order[key]:
                model_name = file
                print(f"converting {model_name}....")
                if "TorchQAT" in model_name:
                    jit_model = torch.jit.load(model_name)
                    _,int8_acc = Evaluating(jit_model, test_loader,"cpu")
                    print(f"{int8_acc:.2f}acc",end="\n\n")
                    data = pd.DataFrame([model_name,int8_acc])
                    data.to_csv(csv_name, mode='a',header=False,index=False)
                else:
                    NEW_MODEL.load_state_dict(torch.load(model_name))
                    _, val_acc = Evaluating(NEW_MODEL, test_loader,device=cpu_device)
                    print(f"Before Model acc : {val_acc:.2f}%")
                    
                    for i in range(kargs["loop"]):
                        print(f"loop {i+1}")
                        add = [model_name,val_acc]
                        for q in qconfig:
                            default_q = copy.deepcopy(NEW_MODEL)
                            quantize_model(default_q, data= train_loader,qconfig=q)
                            _,int8_acc = Evaluating(default_q,test_loader,"cpu")
                            print(f"--> {int8_acc:.2f}acc",end="\n\n")
                            add.append(f"{int8_acc:.2f}")
                        data = pd.DataFrame([add])
                        if kargs["savename"]:
                            data.to_csv(csv_name, mode='a',header=False,index=False)
        