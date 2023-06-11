import torch
from torch import nn


class ConvUnit(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, conv_kernel: int = 3, pool_kernel: int = 2, normalization: bool = True) -> None:
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, stride=1, padding=1)
    self.relu = nn.ReLU()
    self.normalize = normalization
    if normalization:
      self.batch_norm = nn.BatchNorm2d(out_channels)
    self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv(x)
    x = self.relu(x)
    if self.normalize:
      x = self.batch_norm(x)
    x = self.pool(x)
    return x


class DenseUnit(nn.Module):
  def __init__(self, in_features: int, out_features: int, dropout: int = 0, normalization: bool = True) -> None:
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)
    self.relu = nn.ReLU()
    self.normalize = normalization
    if normalization:
      self.batch_norm = nn.BatchNorm1d(out_features)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.linear(x)
    x = self.relu(x)
    if self.normalize:
      x = self.batch_norm(x)
    x = self.dropout(x)
    return x


from torch import nn

class HeadgearRecognizer(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.extractor = nn.Sequential(
      ConvUnit(3, 32, conv_kernel=5, pool_kernel=3),
      ConvUnit(32, 64),
      ConvUnit(64, 128),
      ConvUnit(128, 256),
    )
    self.flatten = nn.Flatten()
    self.classifier = nn.Sequential(
      DenseUnit(20736, 1024),
      DenseUnit(1024, 512, dropout=0.25),
      DenseUnit(512, 256, dropout=0.25),
      DenseUnit(256, 20, normalization=False),
    )
    self.softmax = nn.LogSoftmax(dim=1)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.extractor(x)
    x = self.flatten(x)
    x = self.classifier(x)
    return self.softmax(x)
