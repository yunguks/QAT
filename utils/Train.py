import torch
from tqdm import tqdm
import numpy as np
from .Check import memory_check
import copy

__all__ = ["Training", "Evaluating", "custom_quant_weights","custom_dequant_weights","fake_weight_quant"]
    
def Training(model, train_loader, test_loader, device, optimizer, scheduler, epochs=100,save_name=None,fake=False):
    """_summary_

    Args:
        model (module): Pytorch model
        train_loader (require): Train_Loader
        test_loader (require): Test_Loader
        device (require): Device
        optimizer (require): Optimizer
        scheduler (require): Scheduler
        epochs (int, optional): . Defaults to 100.
        save_name (str, optional): Best Loss model file name. Defaults to "test".
        fake (bool,optional) : fake weight quantization before forward. Dafualt False

    Returns:
        model
    """
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    print("Before Training")
    torch.cuda.memory_reserved()
    memory_check()
    count = 0
    best_loss = np.Inf
    # Training
    model.to(device)
    for epoch in range(epochs):

        running_loss = 0
        running_corrects = 0
        model.train()

        for inputs, labels in tqdm(iter(train_loader),leave=False):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if fake:
                model = fake_weight_quant(model)
            outputs = model(inputs)

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
        val_loss, val_acc = Evaluating(model,test_loader,device=device,criterion=criterion)
        print(f"--------{epoch+1}----------")
        print(f"Train {train_loss:.4f} Loss, {train_accuracy:.2f} Acc")
        print(f"Validation {val_loss:.4f} Loss, {val_acc:.2f} Acc")

        if best_loss > val_loss:
            best_loss = val_loss
            count = 0
            if save_name is None:
                save_name = "./models/weights/test.pt"
            torch.save(model.state_dict(),save_name)
        else:
            count +=1
            if count > 10:
                break
    model.load_state_dict(torch.load(save_name)) 
    return model

def Evaluating(model, test_loader, device, criterion=None):
    """_summary_

    Args:
        model (required): _description_
        test_loader (required): _description_
        device (required): _description_
        criterion (optional): Defaults to None.

    Returns:
        Loss, Acc: Validation Score 
    """
    model.to(device)
    model.eval()

    running_loss = 0
    running_corrects = 0

    for inputs, labels in tqdm(iter(test_loader),leave=False):
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0
        # statistics
        running_loss += loss * labels.size(0)
        running_corrects += (preds == labels).sum().item()

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = 100 * running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

def custom_quant_weights(model:torch.nn.Module):
    """_summary_
    Quantize model weights [-0.127,0.127]
    Args:
        model (required): 

    Returns:
        model, [M,n] list
    """

    backup = []
    for name, param in model.named_parameters():
        temp = param.data.clone()
        M = torch.max(temp)
        m = torch.min(temp)
        backup.append([m,M])
        # param.data = param.data.clamp(m,M)
        param.data = torch.round(254*(param.data-m)/(M-m)-127)/1000

    return model,backup

def custom_dequant_weights(model:torch.nn.Module,backup):
    """_summary_
    Quantize model weights [-0.127,0.127]
    Args:
        model (required): 
        backup (list) : quant M,n list

    Returns:
        model 
    """
    for i, params in enumerate(model.named_parameters()):
        name, param = params
        M = backup[i][1]
        m = backup[i][0]
        param.data = (1000*param.data+127)*(M-m)/254+m

    return model

def fake_weight_quant(model:torch.nn.Module,per_channel=True):
    """_summary_
    Args:
        model (torch.nn.Module): model
        per_channel (bool, optional): Defaults to True. IF True, using quant-dequant with per-channel
    Returns:
        model
    """
    if per_channel:
        min_val_cut = torch.tensor(-0.127)
        max_val_cut = torch.tensor(0.127)
        
        for name, param in model.named_parameters():
            if param.dim()==4:
                y = param.detach()

                Min,Max = torch.aminmax(y, dim=1)
                Min = torch.min(Min,min_val_cut)
                Max = torch.max(Max,max_val_cut)

                s = (Max-Min)/(max_val_cut-min_val_cut)
                z = min_val_cut -torch.round(Min/s,decimals=3)

                for i in range(y.size()[0]):
                    y[i] = torch.round(y[i]/s[i]+z[i],decimals=3)
                    y[i] = (y[i]-z[i])*s[i]
                param=y
                del y,s,z

    else:
        model, backup = custom_quant_weights(model)
        model = custom_dequant_weights(model,backup)
    
    return model