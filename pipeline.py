import torch
import torchvision
import numpy as np
import random
import wandb
import pylab as pl
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import  DataLoader, Subset
from torchmetrics.functional.regression import mean_squared_error, mean_absolute_error
from torch.optim import Adam
from IPython import display



TRAINING_SIZE = 30
TEST_SIZE = 15
IMAGENET_NORMALIZATION = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
TRANSFORM = transforms.Compose([transforms.ToTensor()])
NORMALIZATION_1 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

def data_download(type_normalization, resize):
    if(type_normalization == "imagenet"):
        transform = IMAGENET_NORMALIZATION
    elif(type_normalization== "transform"):
        transform = TRANSFORM
    else:
        transform = NORMALIZATION_1
    if(resize):
        resize_transform = transforms.Resize((224,224))
        transform = transforms.Compose([resize_transform, *transform.transforms])
    training_set = CIFAR10(root='./data',train=True,download=True,transform=transform)
    test_set = CIFAR10(root='./data',train=False,download=True,transform=transform)
    return training_set, test_set

def dataloaders(training_set, test_set):
    train_dl = DataLoader(training_set, batch_size=TRAINING_SIZE,shuffle=True, num_workers=2)
    test_dl = DataLoader(test_set, batch_size=TEST_SIZE,shuffle=False, num_workers=2)
    return train_dl, test_dl


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def wandb_initialization():
    wandb.init(
        # set the wandb project where this run will be logged
        project="ImageReconstruction",

        # track hyperparameters and run metadata
        config={
        "architecture": "VGG19",
        "dataset": "CIFAR-10",
        "epochs": 100,
        }
    )

def traintestpipeline(epochs: int, optim: Adam, criterion: nn.MSELoss, name: str, train_dl:DataLoader, test_dl: DataLoader, model, device):
    wandb_initialization()
    t_loss = np.zeros((epochs))
    v_loss = np.zeros((epochs))
    min_loss = 100 
    pbar = tqdm(range(1, epochs+1)) # tdqm permet tenir text dinàmic
    for epoch in pbar:
        
        train_loss = 0 
        val_loss = 0  
        
        model.train()                                                  
        for (input_img, target) in tqdm(train_dl):
            input_img = input_img.to(device).float()
            # forward pass
            output = model(input_img)

            loss = criterion(output,input_img)
            loss.backward()                                            
            optim.step()                                               
            optim.zero_grad()     
            
            train_loss += loss.item()   
                                                        
        model.eval()   
        with torch.no_grad():                                          
             for (input_img, target) in tqdm(test_dl):
                input_img = input_img.to(device).float()
                # forward pass
                output = model(input_img)
   
                loss = criterion(output, input_img)   
                v_acc = mean_squared_error(output, input_img)
                val_loss += loss.item()  
        
        # RESULTATS
        train_loss /= len(train_dl)
        t_loss[epoch-1] = train_loss
        
        val_loss /= len(test_dl)   
        v_loss[epoch-1] = val_loss
        if(v_loss[epoch-1]<min_loss):
            min_loss=v_loss[epoch-1]
            torch.save(model.state_dict(), name)  
        
        wandb.log({ "Trainingloss": train_loss, "Validationloss": val_loss})
        # VISUALITZACIO DINAMICA
        plt.figure(figsize=(12, 4))
        pl.plot(t_loss[:epoch], label="train")
        pl.plot(v_loss[:epoch], label="validation")
        pl.legend()
        pl.xlim(0, epochs)
        pl.xticks(range(0,epochs,1),range(1,epochs+1,1))
        
        display.clear_output(wait=True)
        display.display(pl.gcf())
        plt.close()

        pbar.set_description(f"Epoch:{epoch} Training Loss:{train_loss} Validation Loss:{val_loss}")
    wandb.finish()

def showImages(test_dl, model, device):
    with torch.no_grad():
        model.eval()
        for data, _ in test_dl:  # Get one batch from the test DataLoader
                data = data.to(device).float()
                output = model(data)
                fig, ax = plt.subplots(1, 2, figsize=(8,4))
                ax[0].imshow(data[0].cpu().permute(1, 2, 0).numpy(), cmap='gray')
                ax[1].imshow(output[0].cpu().permute(1, 2, 0).numpy(), cmap='gray')
                ax[0].set_title("Origina Image")
                ax[1].set_title("Reconstructed Image")
                plt.show()
                break 
        
def assignPretrainedModel(device,name:str, model):
    mmodel =  model.to(device)
    mmodel.load_state_dict(torch.load(name))
    mmodel.eval()
    return mmodel

def calculate_metrics(model, test_dl, device):
    model.eval()
    total_mse = 0 
    total_mae = 0 

    total = len(test_dl)
    

    with torch.no_grad():
        for input_img, _ in test_dl:
            input_img = input_img.to(device).float()

            output = model(input_img)
            mse = mean_squared_error(output, input_img)
            total_mse += mse
            mae = mean_absolute_error(output, input_img) 
            total_mae += mae
    total_mse /=total
    total_mae /=total

    print("MSE:", total_mse)
    print("MAE:", total_mae)