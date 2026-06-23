import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from aerosonar.config import load_default_config

config = load_default_config()

metadata_path = config["paths"]["metadata_csv"]
print(metadata_path)


# class SpectrogramCNN(nn.Module):
#     def __init__(self):
#         super(SpectrogramCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, 3, padding = 1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, (1,5), padding = (0,2))
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 256, 3, padding = 1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.conv4 = nn.Conv2d(256, 512, 3, padding = 1)
#         self.drop_conv = nn.Dropout2d(0.2)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.relu = nn.ReLU()
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(512, 128)
#         self.fc2 = nn.Linear(128, 32)
#         self.fc3 = nn.Linear(32, 2)

#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))
#         x = self.drop_conv(x)
        
#         x = self.global_pool(x)
#         x = torch.flatten(x, 1)

#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x



# class SpectrogramCNN(nn.Module):
#     def __init__(self):
#         super(SpectrogramCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
#         self.bn1 = nn.GroupNorm(num_groups=8, num_channels=32)#nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
#         self.bn2 = nn.GroupNorm(num_groups=8, num_channels=64)#nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
#         self.bn3 = nn.GroupNorm(num_groups=8, num_channels=128)#nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)
#         self.bn4 = nn.GroupNorm(num_groups=8, num_channels=256)#nn.BatchNorm2d(256)
#         self.drop_conv = nn.Dropout2d(0.2)
#         self.pool_freq = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.relu = nn.ReLU()
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(256, 64)
#         self.fc2 = nn.Linear(64, 2)
        
#     def forward(self, x):
#         x = self.pool_freq(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool_freq(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))      
#         x = self.drop_conv(x)  
#         x = self.global_pool(x)
#         x = torch.flatten(x, 1)

#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x =self.fc2(x)
#         return x










class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        # Deeper feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # (64 freq, T/2)

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # (32 freq, T/4)

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # (16 freq, T/8)

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1) # Global context
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)