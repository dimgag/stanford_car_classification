import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import torch.nn as nn 
from torchvision import models
import torch.optim as optim
import torch.nn.functional as F
from torch.optiom import lr_scheduler
from torchsummary import summary


## Data
# Load the dataset
train_dir = "car_data/car_data/train"
valid_dir = "car_data/car_data/test"

# Show images from train directory
train_images = os.listdir(train_dir)
print("Number of images in train directory: ", len(train_images))
print("Sample images from train directory: ", train_images[:5])

# Show images from validation directory
valid_images = os.listdir(valid_dir)
print("Number of images in valid directory: ", len(valid_images))
print("Sample images from valid directory: ", valid_images[:5])


# Data Augmentation
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Required constants.
image_size = 224
batch_size = 32
num_workers = 4

# Training Transforms
def get_train_transform(image_size):
  train_transform = transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(35),
      transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
      transforms.RandomGrayscale(p=0.5),
      transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
      transforms.RandomPosterize(bits=2, p=0.5),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225]
          )
  ])
  return train_transform


def get_valid_transform(image_size):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return valid_transform

def get_datasets():
    """
    Function to prepare the Datasets.
    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        train_dir, 
        transform=(get_train_transform(image_size))
    )
    dataset_valid = datasets.ImageFolder(
        valid_dir, 
        transform=(get_valid_transform(image_size))
    )
    return dataset_train, dataset_valid, dataset_train.classes


def get_data_loaders(dataset_train, dataset_valid):
    """
    Input: the training and validation data.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader 



# Load the training and validation datasets.
dataset_train, dataset_valid, dataset_classes = get_datasets()


# Load the training and validation data loaders
train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)

## Load the model
# Load the EfficientNet-b0 model
model = models.efficientnet_b0(pretrained=True)

# Print the model summary
# summary(model, (3, 224, 224))

# Print the model parameters 
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_trainable_params:,}")

# Freeze the main model 
for param in model.parameters():
	param.requires_grad = True

model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)


# # # # # # # # # # # # # # # # # # # # # 
# create an optimizer
# Send the model to device
model = model.to(device)


# swa_model = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 300)
# optimizer_ft = optim.SGD(parames_to_update, lr=0.001, momentum=0.9)



# Training the model
from tqdm.auto import tqdm

def train(model, trainloader, optimizer, criterion):
  model.train()
  print("Training model...")
  train_running_loss = 0.0
  train_running_correct = 0
  counter = 0

  for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
      counter += 1
      image, labels = data
      image = image.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      # Forward pass.
      outputs = model(image)
      # Calculate the loss.
      loss = criterion(outputs, labels)
      train_running_loss += loss.item()
      # Calculate the accuracy.
      _, preds = torch.max(outputs.data, 1)
      train_running_correct += (preds == labels).sum().item()
      # Backpropagation.
      loss.backward()
      # Update the weights.
      optimizer.step()

  # Loss and accuracy for the complete epoch.
  epoch_loss = train_running_loss / counter
  epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
  return epoch_loss, epoch_acc 




# Validation of the model
def validate(model, testloader, criterion, class_names):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc




import time
epochs = 30

# Start the training.

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                            optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                criterion, dataset_classes)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)
    time.sleep(2)





