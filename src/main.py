import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn 
from torchvision import models
import torch.optim as optim
import torch.nn.functional as F
# from torch.optiom import lr_scheduler
# from torchsummary import summary
import time
# Modules
from dataset import get_datasets
from dataset import get_data_loaders
from model import build_model, get_model_params
from train import train, validate
from utils import save_model, save_plots

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

if __name__ == "__main__":
    ## Dataset
    # Load the dataset directories
    train_dir = "car_data/car_data/train"
    valid_dir = "car_data/car_data/test"

    # train_dir = "/Users/dim__gag/Desktop/stanford-cars-dataset/data/car_data/car_data/train"
    # valid_dir = "/Users/dim__gag/Desktop/stanford-cars-dataset/data/car_data/car_data/test"

    # Show images from train directory
    train_images = os.listdir(train_dir)
    valid_images = os.listdir(valid_dir)

    # Required constants.
    image_size = 224
    batch_size = 32
    num_workers = 4
    epochs = 100

    ## Data Augmentation
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    # Load the training and validation data loaders
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)



    ## Get the fine tuned model
    model = build_model()
    # Show the model parameters
    # get_model_params(model)


    # # Define the device.
    # device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Computation device: {device}")
    # # Add the model to the device.
    # model = model.to(device)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    ## OPTIMIZER
    # EXPERIMENTS WITH OTHER OPTIMISERS AND OTHER LEARNING RATES CAN BE DONE HERE
    
    # optimizer = optim.Adam(model.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    
    # LEARNING RATE SCHEDULER
    # Step-wise Learning Rate Decay
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Reduce on Loss Plateau Decay
    # Reduce on Loss Plateau Decay, Patience=0, Factor=0.1

    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)
    
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # Reduce on Loss Plateau Decay, Patience=0, Factor=0.5 # NEXT EXPERIMENT
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)
          
    # Then do all the experiments with SGD Optimizer " optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True) "
            
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    
    ## Train the model
    # Lists to keep track of the loss and accuracy.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
    for epoch in range(epochs):
        # Print Num Epoch
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                    criterion, dataset_classes)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        
        # Decay Learning Rate
        scheduler.step()
        
        # Decay lr_scheduler.ReduceLROnPlateau
        # scheduler.step(valid_epoch_acc)
        
        # Print Learning Rate
        print('LR:', scheduler.get_lr())
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        time.sleep(2)


    ## Save the model and Plot the loss and accuracy curves.
    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    print('TRAINING COMPLETE')
