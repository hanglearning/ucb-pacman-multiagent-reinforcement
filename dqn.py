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
        self.activation1 = nn.Sigmoid()
        self.fc2 = nn.Linear(16, 8)
        self.activation2 = nn.Sigmoid()
        self.dropout = nn.Dropout(.2)
        self.fc3 = nn.Linear(8, 1)
        self.criterion = nn.MSELoss()
    
    @staticmethod
    def dict2vec(counter):
        features = []
        dict_key = sorted(counter.keys())
        for feat in dict_key:
            features.append(counter[feat])
        return np.array(features).astype(np.float32)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        # x = self.dropout(x)
        o = self.fc3(x)
        return o