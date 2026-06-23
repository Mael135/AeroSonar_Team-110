import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
df = pd.read_csv("drones.csv")
df.set_index('fname',inplace=True)
for f in df.index:
    rate, signal = wavfile.read('wavfiles/' + f)
    df.at[f, 'length'] = signal.shape[0]/rate
classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()
fig, ax = plt.subplots()


class SpectrogramCNN(nn.modules):
    def __init__(self, freq_bins , time_frames, num_classes):
        super(SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_chanels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_chanels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.drop1 = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=32, out_chanels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_chanels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.drop2 = nn.Dropout(p=0.25)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64) 
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, x):
        # block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        # block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        # global avrage pooling
        x = self.global_avg_pool(x)
        # flatten (batch size, 128)
        x = x.view(x.size(0),-1)
        # classification
        x = F.relu(self.fc1)
        x = self.drop3(x)
        x = self.fc2(x)
        return x










