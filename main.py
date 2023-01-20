from models import quat_mobilenet_v2
from models import Quant_ReLU
from torchsummary import summary
from utils import Data
from utils import Train
import torch

if __name__=="__main__":
    # device 
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    
    # model load
    NEW_MODEL = quat_mobilenet_v2(cifar10=True,activation_layer = Quant_ReLU)
    summary(NEW_MODEL,(3,32,32),device="cpu")

    # data load
    train_dataloader, test_loader = Data.Cifar10_Dataloader(quantize=True)
    
    # optimizer 
    optimizer = torch.optim.SGD(NEW_MODEL.parameters(),lr=1e-2,momentum=0.9)
    
    # scheduler 
    scheduler = None
    
    # train model
    NEW_MODEL = Train.Training(NEW_MODEL,train_dataloader,test_loader,gpu_device,optimizer,scheduler)
    
