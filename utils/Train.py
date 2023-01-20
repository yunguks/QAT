import torch
from tqdm import tqdm
import numpy as np
from .Check import memory_check
import copy

__all__ = ["Training", "Evaluating", "custom_quant_weights","custom_dequant_weights"]
    
def Training(model, train_loader, test_loader, device, optimizer, scheduler, epochs=100,model_name="test"):
    """_summary_

    Args:
        model (module): Pytorch model
        train_loader (require): Train_Loader
        test_loader (require): Test_Loader
        device (require): Device
        optimizer (require): Optimizer
        scheduler (require): Scheduler
        epochs (int, optional): . Defaults to 100.
        model_name (str, optional): Best Loss model file name. Defaults to "test".

    Returns:
        model
    """
    

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

def custom_quant_weights(model,device="cpu"):
    """_summary_
    Quantize model weights [-0.127,0.127]
    Args:
        model (required): 
        inplace (bool, optional): . Defaults to True. If not, return new_model 

    Returns:
        model, [M,n] list
    """
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        state = model.state_dict()
        backup = []
    
        total_tensor = torch.tensor([])
        for i in state.keys():
            new_param = state[i].view(-1)
            total_tensor = torch.cat((total_tensor,new_param),0)

        total_tensor,_ = total_tensor.sort()
        number = int(len(total_tensor)*0.05)
        
        M = total_tensor[-number]
        m = total_tensor[number]
        backup.append([M,m])
        for i in state.keys():
            param = state[i]
            new_param = param.clamp(m,M)
            new_param = torch.round(254*(new_param-m)/(M-m)-127)
            new_param = new_param/1000
            # param.data = torch.quantize_per_tensor(new_param, 0.1, 10, torch.quint8)
            state[i] = new_param
        
        model.load_state_dict(state)
        model.to(device)
        model.train()
    return model,backup

def custom_dequant_weights(model,backup,device="cpu"):
    """_summary_
    Quantize model weights [-0.127,0.127]
    Args:
        model (required): 
        backup (list) : quant M,n list
        inplace (bool, optional): . Defaults to True. If not, return new_model 

    Returns:
        model 
    """
    with torch.no_grad():
        state = model.state_dict()

        max=backup[0][0]
        min=backup[0][1]
        for i,k in enumerate(state.keys()):
            new_param = state[k]
            new_param = (1000*new_param+127)*(max-min)/254+min
            # param.data = torch.quantize_per_tensor(new_param, 0.1, 10, torch.quint8)
            state[k] = new_param
        
        model.load_state_dict(state)
        model.to(device)
        model.train()
    return model

    