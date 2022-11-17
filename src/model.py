import torch.nn as nn
from torchvision import models

def get_model_params(model):
    """Get model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_trainable_params:,}")


def build_model(pretrained=True, fine_tune=False, num_classes=196):
  if pretrained:
    print("Loading pre-trained weights")
  else:
    print("Not loading pre-trained weights")
  model = models.efficientnet_b0(pretrained=pretrained)

  if fine_tune:
    print("Fine-tuning the model")
    for param in model.parameters():
      param.requires_grad = True
  elif not fine_tune:
    print("Not fine-tuning the model")
    for param in model.parameters():
      param.requires_grad = False

  # Classification Head
  model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
  return model
