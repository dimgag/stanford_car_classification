from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

image_size = 224
batch_size = 32
num_workers = 4

## Data directoris
# Local Paths
train_dir = "data/stanford-cars-dataset/data/car_data/car_data/train"
valid_dir = "data/stanford-cars-dataset/data/car_data/car_data/train"

train_images = os.listdir(train_dir)
valid_images = os.listdir(valid_dir)

## Data Augmentation
# Training Data Transforms
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

# Validation Data Transforms
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
