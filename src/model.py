import torch.nn as nn
from torchvision import models

def build_model(pretrained=True, fine_tune=True, num_classes=10):
    if pretrained:
        print("Loading pre-trained weights")
    else:
        print("Not loading pre-trained weights")
    model = models.efficientnet_b1(pretrained=pretrained)

    if fine_tune:
        print("Fine-tuning the model")
        for param in model.parameters():
            param.requires_grad = True
    elif not fine_tune:
        print("Not fine-tuning the model")
        for param in model.parameters():
            param.requires_grad = False
    
    # Change the Classification Head
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model