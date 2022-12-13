# Stanford Car Classification
This is a project for the Stanford Car Classification Challenge on Kaggle. 
The goal is to classify cars into 196 classes. 

## 💾 Folder data
The dataset contains 16,185 car images distributed over 196 classes/brands. There are 8,144 images for training and 8,041 images for testing in this dataset. Each class roughly has a 50-50 split in the training and validation set. The dataset is available [here](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder).

## 🚀 Folder models
This folder contains results observed from transfer-learning & fine-tuning the EfficientNet-b0 model on the Stanford Car dataset.

The models are trained using the following hyperparameters:
| Classification Head |  Optimizer | Learning Rate | Accuracy | Loss |
| --- | --- | --- | --- | --- |
| Classification Head 1 | Adam |   lr=0.001 | 0.81   | 0.9 |
| Classification Head 2 | Adam |   lr=0.001 | 0.81   | 0.8  |
| Classification Head 3 | Adam |   lr=0.001 | 0.815  | 0.8  |
| Classification Head 3 | SGD  |   lr=0.001 | 0.82   | 0.6  |
| Classification Head 3 | Adam |   StepLR (Decay every 1 epoch)   | - | -   |
| Classification Head 3 | Adam |   StepLR (Decay every 10 epoch)  | - | -   |
| Classification Head 3 | Adam |   ReduceLROnPlateau (factor=0.1) | - | -   |
| Classification Head 3 | SGD  |  StepLR (Decay every 10 epoch)   | 0.67 | 1.2 |
| Classification Head 3 | SGD  |  ReduceLROnPlateau (factor=0.1)  | 0.75 | 0.8 |
| Classification Head 3 | SGD  |  ReduceLROnPlateau (factor=0.5)  | 0.84 | 0.5 |



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
| [`dataset.py`](https://github.com/dimgag/stanford_car_classification/blob/master/src/dataset.py) | Contains functions for loading and preprocess the data.|
| [`model.py`](https://github.com/dimgag/stanford_car_classification/blob/master/src/model.py) | Contains the functions that define the model. |
| [`train.py`](https://github.com/dimgag/stanford_car_classification/blob/master/src/train.py) | Contains the training and validation functions.|
| [`utils.py`](https://github.com/dimgag/stanford_car_classification/blob/master/src/utils.py) | Contains utility functions, such as plots and model saving.|
| [`main.py`](https://github.com/dimgag/stanford_car_classification/blob/master/src/main.py) | Contains the main function that runs the training. |



## 📝 File requirements.txt
Contains the list of dependencies for the project.
```bash
pip install -r requirements.txt
```