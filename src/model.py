import torch
import torch.nn as nn
from torchvision import models


# Device - GPU or CPU
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Computation device: {device}")

def get_model_params(model):
  """Get model parameters"""
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total parameters: {total_params:,}")
  total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Trainable parameters: {total_trainable_params:,}")



def build_model(pretrained=True, freeze=False, num_classes=196):
  torch.manual_seed(42)
  torch.cuda.manual_seed(42)
  # Model Weights - Load Pretrained / Not Pretrained model
  if pretrained:
    weights = models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights
    model = models.efficientnet_b0(weights=weights).to(device)
  else:
    model = models.efficientnet_b0(weights=None).to(device)

  print("-"*50)
  print("Original model parameters:")
  get_model_params(model)
  print("-"*50)
  
  # Freeze the model
  if freeze:
    print("Freezing the base model")
    for param in model.features.parameters():
      param.requires_grad = False
  elif not freeze:
    print("Not freezing the base model")
    for param in model.features.parameters():
      param.requires_grad = True
  
  print("\nAdding Classification Head . . .")
    # Add the classification head    # New test - 3 layer     
  model.classifier = nn.Sequential(
    nn.Linear(in_features=1280, out_features=512),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Softmax(dim=1),
    nn.Linear(in_features=256, out_features=num_classes)).to(device)


  print("-"*50)
  print("New model parameters:")
  get_model_params(model)
  print("-"*50)

  return model    

    
#     Results 5
  # Add multiclass classification head
  # model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes).to(device)
        
    
#     Results 4 - good results
  # model.classifier = torch.nn.Sequential(
  #   torch.nn.Linear(in_features=1280, out_features=640, bias=True),
  #   torch.nn.Dropout(p=0.2, inplace=True),
  #   torch.nn.Linear(in_features=640, out_features=320, bias=True),
  #   torch.nn.Dropout(p=0.2, inplace=True),
  #   torch.nn.Linear(in_features=320, out_features=num_classes, bias=True)).to(device)


# # Results 6 - Next one maybe
#   model.classifier = nn.Sequential(
#       torch.nn.Linear(in_features=1280, out_features=640, bias=True),
#       nn.AdaptiveAvgPool2d(320),
#       nn.Dropout(p=0.2, inplace=True),
#       torch.nn.Linear(in_features=320, out_features=num_classes, bias=True)).to(device)


  


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# from torchinfo import summary

# model = build_model()

# summary(model, 
#         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
#         verbose=0,
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )

