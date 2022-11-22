import torch
import matplotlib.pyplot as plt
import os 

from dataset import get_valid_transform, get_datasets, get_data_loaders


# Required constants.
image_size = 224
batch_size = 32
num_workers = 4

## Load the Dataset
train_dir = "/Users/dim__gag/Desktop/stanford-cars-dataset/data/car_data/car_data/train"
valid_dir = "/Users/dim__gag/Desktop/stanford-cars-dataset/data/car_data/car_data/test"

train_images = os.listdir(train_dir)
valid_images = os.listdir(valid_dir)

## Data Augmentation
# Load the training and validation datasets.
dataset_train, dataset_valid, dataset_classes = get_datasets()
# Load the training and validation data loaders
train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)



## Load the model
model_dir = '/Users/dim__gag/Desktop/results3-classifier3-waiting/model.pth'

# Device - GPU or CPU
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}")

# Load the model
model = torch.load(model_dir, map_location=torch.device('cpu'))


torch.manual_seed(42)
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    
    loss, acc = 0, 0
    # model.eval() # Got error:AttributeError: 'dict' object has no attribute 'eval'
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}




# Accuracy
def accuracy(y_true, y_pred):
    """
    Calculates accuracy.
    """
    correct_preds = torch.sum(y_true == y_pred)
    return correct_preds.float() / len(y_true)
    

import torch.nn as nn



eval_model(model=model,
              data_loader=valid_loader,
                loss_fn=nn.CrossEntropyLoss(),
                accuracy_fn=accuracy)

