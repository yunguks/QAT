from torchvision import transforms
import os 
import torchvision
import torch
import numpy as np

def Cifar10_Dataloader(quantize=False,only_dataset=False, batch_size:int=128,filepath=None):
    """_summary_

    Args:
        Quantize (bool, optional): Defaults to False. If True, input scale = [-0.127,0.127]
        only_dataset (bool, optional): Defaults to False. If True, return [train_dataset,test_dataset] not loader
        batch_size (int, optional): _description_. Defaults to 128.

    Returns:
        DataLoader: train_loader, test_loader
    """
    if quantize:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            input_quant(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            input_quant(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if filepath is None:
        filepath = "data"
    
    train_dataset = torchvision.datasets.CIFAR10(root=filepath, train=True, download=True, transform=train_transform) 
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_dataset = torchvision.datasets.CIFAR10(root=filepath, train=False, download=True, transform=test_transform)
    print(f"Train data set = {len(train_dataset)}, Test = {len(test_dataset)}")
    if only_dataset:
        return train_dataset, test_dataset

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        sampler=test_sampler)
    return train_loader, test_loader

import wget
def ImageNet_DataLoader(split_num = [0.08,0.02,0.9]):
    """_summary_
    If you don't have ImageNet data, you can download ImageNet Validation Data using wget
    
    Args:
        split_num (list, optional): _description_. Defaults to [0.08,0.02,0.9].
        split 60,000 ImageNet Validation Data. need 3 value

    Returns:
        DataLoader: train_loader, test_loader
    """
    if not os.path.exists("./data/ImageNet/meta.bin"):
        print("Meta data download")
        wget.download(url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz", out="./data/ImageNet")
    # if not os.path.exists("./data/ImageNet/ILSVRC2012_devkit_t3.tar.gz"):
    #     print("Toolkit t3 Download")
    #     toolkit_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t3.tar.gz"
    #     wget.download(url= toolkit_url,out="./data/ImageNet")
    if not os.path.exists("./data/ImageNet/ILSVRC2012_img_val.tar"):
        print("Download val data")
        val_url  = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar'
        wget.download(url=val_url, out="./data/ImageNet")

    # if not os.path.exists("./data/ImageNet/ILSVRC2012_img_train_t3.tar"):
    #     print("Download train t3 data")
    #     train_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar"
    #     wget.download(url=train_url,out="./data/ImageNet")
    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    print(os.getcwd())
    dataset = torchvision.datasets.ImageNet(root="./data/ImageNet",split="val", transform = train_transform)
    Train_dataset, Test_dataset,_ = torch.utils.data.random_split(dataset, split_num)
    print(f"Train data set = {len(Train_dataset)}, Test = {len(Test_dataset)}")
    
    train_sampler = torch.utils.data.RandomSampler(Train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(Test_dataset)

    Train_loader = torch.utils.data.DataLoader(dataset=Train_dataset, batch_size= 32, sampler = train_sampler)
    Test_loader = torch.utils.data.DataLoader(dataset=Test_dataset, batch_size =32, sampler = test_sampler)
    return Train_loader, Test_loader

class input_quant(object):
    """_summary_
        Convert a ``PIL Image`` to tensor.
        Converts a PIL Image (H x W x C) in the range
        [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [-0.127, 0.127]
    """
    
    def __call__(self,data):
        data = data/1000
        return data