# Stanford Car Classification
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-orange.svg)](https://pytorch.org/) 
[![torchvision](https://img.shields.io/badge/torchvision-0.13.1-orange.svg)](https://pytorch.org/)

This is a project for the Stanford Car Classification Challenge on Kaggle. 
The goal is to classify cars into 196 classes. 

## 💾 Folder data
The dataset contains 16,185 car images distributed over 196 classes/brands. There are 8,144 images for training and 8,041 images for testing in this dataset. Each class roughly has a 50-50 split in the training and validation set. The dataset is available [here](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder).

## 🚀 Folder models
This folder contains results observed from transfer-learning & fine-tuning the EfficientNet-b0 model on the Stanford Car dataset.

The models are trained using the following hyperparameters:
| Model | Description | Optimizer | Learning Rate | Batch Size | Epochs | Accuracy | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Model-1]() | Transfer Learning | Adam | 0.001 | 32 | 50 | 0.85 | Completed |
| [Model-2]() | Classification Head 1 | Adam | 0.001 | 32 | 50 | 0.90 | Completed |
| [Model-3]() | Classification Head 2 | Adam | 0.001 | 32 | 50 | 0.91 | Completed |
| [Model-4]() | Classification Head 3 | Adam | 0.001 | 32 | 50 | 0.91 | Training |
| [Model-5]() | Classification Head 3 | SGD | 0.001 | 32 | 50 | 0.91 | ToDo |
| [Model-6]() | Classification Head 3 | Adam | Lr_scheduler | 32 | 50 | 0.91 | ToDo |
| [Model-7]() | Classification Head 3 | SGD | Lr_scheduler | 32 | 50 | 0.91 | ToDo |

### Classification Heads
The classification heads are the last layers of the model. The classification heads are trained on the top of the pre-trained model.
```python 
# Classficication Head 1
model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes).to(device)

# Classficication Head 2
model.classifier = torch.nn.Sequential(
  torch.nn.Linear(in_features=1280, out_features=640, bias=True),
  torch.nn.Dropout(p=0.2, inplace=True),
  torch.nn.Linear(in_features=640, out_features=320, bias=True),
  torch.nn.Dropout(p=0.2, inplace=True),
  torch.nn.Linear(in_features=320, out_features=num_classes, bias=True)).to(device)

# Classficication Head 3
model.classifier = nn.Sequential(
  nn.Linear(in_features=1280, out_features=640),
  nn.Linear(in_features=640, out_features=320),
  nn.Dropout(0.5),
  nn.Linear(in_features=320, out_features=num_classes)).to(device)

```


## 🔌 Folder src   
Contains the source code for the project.
| File | Description |
| --- | --- |
| [`dataset.py`](https://github.com/dimgag/stanford_car_classification/blob/master/src/dataset.py) | Contains the `Data` class that loads the data and preprocesses it. |
| [`eval.py`](https://github.com/dimgag/stanford_car_classification/blob/master/src/eval.py) | Contains the `Evaluator` class that evaluates the model. |
| [`model.py`](https://github.com/dimgag/stanford_car_classification/blob/master/src/model.py) | Contains the `Model` class that defines the model. |
| [`train.py`](https://github.com/dimgag/stanford_car_classification/blob/master/src/train.py) | Contains the `Trainer` class that trains the model. |
| [`utils.py`](https://github.com/dimgag/stanford_car_classification/blob/master/src/utils.py) | Contains utility functions, such as Accuracy & Loss plot functions.|



## 📝 File requirements.txt
Contains the list of dependencies for the project.
```bash
pip install -r requirements.txt
```

1. Unzip files in data folder

unzip data/stanford-cars-dataset/data.zip

2. Install dependences 

pip install -r requirments.txt
