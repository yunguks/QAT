from models import quat_mobilenet_v2, MobileNet_V2_Weights, quantize_model
from models import Quant_ReLU,Quant_ReLU6,replace_Qrelu,replace_relu
from torchsummary import summary
from utils import Data,set_random_seeds
from utils.Train import Training, Evaluating, fake_weight_quant
import torch
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

def get_args():
    parser = argparse.ArgumentParser(description="QAT training test")
    # model
    parser.add_argument("--model",type=str,default="mobilenet", help="select model. Default mobilenet")
    parser.add_argument("--tiny", action='store_true', help="select tiny model.")
    parser.add_argument("--weights", type=str, help="load weights file name.")
    parser.add_argument("--pretrain", action="store_true", help="using cifar10 pretrain weights.")
    # train
    parser.add_argument("--device", type=int, default=None,help="select device number")
    parser.add_argument("--lr", type=float,default=1e-3, help="initial lr. Default 1e-3.")
    parser.add_argument("--seed", type=int,default=42, help="select random seed. Default 42.") 
    parser.add_argument("--epoch", type=int, default=200, help="train epoch. Default 200.")
    # quantization
    parser.add_argument("--act", action='store_true', help="using layer.Quant_ReLU6 with activation output quant.")
    parser.add_argument("--torchQAT", action="store_true", help="using pytorch Quantization Aware Training.")
    parser.add_argument("--input-fake",action="store_true", help="fake quantization for input.")
    parser.add_argument("--weight-fake",action="store_true", help="fake quantization for weight.")
    parser.add_argument("--fake", action="store_true",help="fake quantization both input and weight.")
    # result
    parser.add_argument("--savename", type=str, default=None, help="Name to store the model. Default test.pt.")
    parser.add_argument("--result", action="store_true", help="export log csv.")
    return vars(parser.parse_args())

if __name__=="__main__":
    kargs = get_args()
    print(kargs)

    # set random see
    set_random_seeds(kargs["seed"])
    
    # device 
    if torch.cuda.is_available():
        if kargs["device"]:
            gpu_device = torch.device(f"cuda:{kargs['device']}")
        else:
            gpu_device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    
    # activation quant
    if kargs["act"] or kargs['fake']:
        activation_layer = Quant_ReLU6
    else:
        activation_layer = torch.nn.ReLU6
    
    # model load
    if kargs["tiny"]:
        MODEL = quat_mobilenet_v2(cifar10=True)
        if kargs["pretrain"]:
            MODEL.load_state_dict(torch.load("./models/weights/tiny_mobilenetv2_cifar.pt"))
    else:
        MODEL = quat_mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1,activation_layer=activation_layer)
        MODEL.classifier.append(torch.nn.Dropout(0.2))
        MODEL.classifier.append(torch.nn.Linear(1000, 10))
        if kargs["pretrain"]:
            MODEL.load_state_dict(torch.load("./models/weights/q_mobilenetv2_cifar10.pt"))
        
    if kargs["weights"]:
        MODEL.load_state_dict(torch.load(kargs["weights"]))
    
    # pytorch QAT prepare
    if kargs["torchQAT"]:
        from models import replace_relu
        replace_relu(MODEL)
        MODEL.fuse_model()
        MODEL.train()
        # activation - histogram , weight - perchannelMinMax
        MODEL.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        MODEL = torch.quantization.prepare_qat(MODEL)
    
    # model weight save name and result
    if kargs["savename"] is None:
        save_name = "./result/test.pt"
    else:
        save_name = kargs["savename"]
        if ".pt" not in kargs["savename"]:
            save_name +=".pt"
    if kargs["torchQAT"]:
        save_name = save_name.replace(".pt","_jit.pt")
    
    if os.path.exists(save_name):
        # qat_model = torch.jit.load(save_name)
        # print(qat_model.state_dict())
        MODEL.load_state_dict(torch.load(save_name))
        
    csv_name = save_name.replace(".pt",".csv")
    if os.path.exists(csv_name) is False:
        header = pd.DataFrame({"epoch":[],"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]})
        header.to_csv(csv_name,index=False)
    
    # data load
    train_loader, test_loader = Data.Cifar10_Dataloader()
    
    # optimizer 
    optimizer = torch.optim.SGD(MODEL.parameters(),lr=kargs["lr"],momentum=0.9)
    
    # # scheduler 
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,60,90], gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=kargs["lr"]*0.001)
    # # train model
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)


    count = 0
    best_loss = np.Inf
    # Training
    MODEL.to(gpu_device)
    for epoch in range(kargs["epoch"]):

        running_loss = 0
        running_corrects = 0
        MODEL.train()

        for inputs, labels in tqdm(iter(train_loader),leave=False):
            inputs = inputs.to(gpu_device)
            labels = labels.to(gpu_device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # input fake
            if kargs["input_fake"] or kargs["fake"]:
                with torch.no_grad():
                    for i in range(inputs.size()[0]):
                        temp = inputs[i].detach()
                        M = torch.max(temp)
                        m = torch.min(temp)
                        inputs[i] = torch.round(254*(inputs[i]-m)/(M-m)-127)/1000
                        inputs[i] = (1000*inputs[i]+127)*(M-m)/254+m
                    
            # forward + backward + optimize
            if kargs["fake"] or kargs["weight_fake"]:
                MODEL = fake_weight_quant(MODEL)
            outputs = MODEL(inputs)

            loss = criterion(outputs, labels)
 
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            # statistics
            running_loss += loss.item() * labels.size(0)
            running_corrects += (preds == labels).sum().item()

        # Set learning rate scheduler
            if scheduler is not None:
                scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * running_corrects / len(train_loader.dataset) 

        # Evaluation
        if kargs["act"] or kargs['fake']:
            replace_relu(MODEL)
        val_loss, val_acc = Evaluating(MODEL,test_loader,device=gpu_device,criterion=criterion)
        if kargs["act"] or kargs['fake']:
            replace_Qrelu(MODEL)
        print(f"--------{epoch+1}----------")
        print(f"Train {train_loss:.4f} Loss, {train_accuracy:.2f} Acc")
        print(f"Validation {val_loss:.4f} Loss, {val_acc:.2f} Acc")

        data = pd.DataFrame({"epoch":[epoch+1],
                             "train_loss":[train_loss],
                             "train_acc":[train_accuracy],
                             "val_loss":[val_loss],
                             "val_acc":[val_acc]
        })
        data.to_csv(csv_name, mode='a',header=False,index=False)
        
        if best_loss > val_loss:
            best_loss = val_loss
            count = 0
            if kargs["torchQAT"]:
                q_model = copy.deepcopy(MODEL)
                q_model.to(cpu_device)
                q_model.eval()
                q_model = torch.ao.quantization.convert(q_model)
                _,int8_acc = Evaluating(q_model,train_loader,cpu_device)
                torch.jit.save(q_model,save_name)
            else:
                torch.save(MODEL.state_dict(),save_name)
            
        else:
            count +=1
            if count > 10:
                break
    MODEL.load_state_dict(torch.load(save_name)) 