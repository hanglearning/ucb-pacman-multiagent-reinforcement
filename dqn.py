import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as func

class DQN(nn.Module):
    def __init__(self, n_features=8):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_features, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 5)
        self.dropout = nn.Dropout(.5)
        self.activation = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc2.weight)
        nn.init.normal_(self.fc3.weight)
        nn.init.normal_(self.fc4.weight)
    
    @staticmethod
    def dict2vec(counter):
        features = []
        dict_key = sorted(counter.keys())
        for feat in dict_key:
            features.append(counter[feat])
        return np.array(features).astype(np.float32)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x