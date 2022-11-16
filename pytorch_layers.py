import torch.nn as nn

# Available CNN layers

# Convolutional layers
Conv1d = nn.Conv1d
Conv2d = nn.Conv2d
Conv3d = nn.Conv3d
ConvTranspose1d = nn.ConvTranspose1d
ConvTranspose2d = nn.ConvTranspose2d
ConvTranspose3d = nn.ConvTranspose3d

# Pooling layers
MaxPool1d = nn.MaxPool1d    
MaxPool2d = nn.MaxPool2d
MaxPool3d = nn.MaxPool3d
AvgPool1d = nn.AvgPool1d
AvgPool2d = nn.AvgPool2d
AvgPool3d = nn.AvgPool3d
AdaptiveMaxPool1d = nn.AdaptiveMaxPool1d
AdaptiveMaxPool2d = nn.AdaptiveMaxPool2d
AdaptiveMaxPool3d = nn.AdaptiveMaxPool3d
AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d
AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d
AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d

# Non-linear activations
ReLU = nn.ReLU
ReLU6 = nn.ReLU6
PReLU = nn.PReLU
LeakyReLU = nn.LeakyReLU
ELU = nn.ELU
SELU = nn.SELU
CELU = nn.CELU
GELU = nn.GELU
Sigmoid = nn.Sigmoid
Tanh = nn.Tanh
Softmax = nn.Softmax
Softmax2d = nn.Softmax2d
LogSoftmax = nn.LogSoftmax

# Normalization layers
BatchNorm1d = nn.BatchNorm1d
BatchNorm2d = nn.BatchNorm2d
BatchNorm3d = nn.BatchNorm3d
InstanceNorm1d = nn.InstanceNorm1d
InstanceNorm2d = nn.InstanceNorm2d
InstanceNorm3d = nn.InstanceNorm3d
LayerNorm = nn.LayerNorm
GroupNorm = nn.GroupNorm
LocalResponseNorm = nn.LocalResponseNorm

# Dropout layers
Dropout = nn.Dropout
Dropout2d = nn.Dropout2d
Dropout3d = nn.Dropout3d

# Linear layers
Linear = nn.Linear
Bilinear = nn.Bilinear
Conv1d = nn.Conv1d
Conv2d = nn.Conv2d
Conv3d = nn.Conv3d
ConvTranspose1d = nn.ConvTranspose1d
ConvTranspose2d = nn.ConvTranspose2d
ConvTranspose3d = nn.ConvTranspose3d

# Recurrent layers
RNN = nn.RNN
LSTM = nn.LSTM
GRU = nn.GRU

# Other layers
Embedding = nn.Embedding
EmbeddingBag = nn.EmbeddingBag
PixelShuffle = nn.PixelShuffle
Upsample = nn.Upsample
UpsamplingBilinear2d = nn.UpsamplingBilinear2d

