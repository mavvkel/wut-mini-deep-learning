import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class TreeLayer(nn.Module):
    """Tree sampling layer"""
    def __init__(self, sizes,activation):
        super().__init__()

        self.sizes = sizes
        self.activation = activation

        weights = torch.empty(self.sizes)
        bias = torch.empty(1,self.sizes[1],self.sizes[3],self.sizes[4])
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)

        if self.activation == "relu":
            nn.init.kaiming_normal_(self.weights)
        elif self.activation == "sigmoid":
            nn.init.normal_(self.weights,mean=0.0,std=1.0)
        nn.init.kaiming_normal_(self.bias)

    def forward(self, x):
        w_times_x = torch.mul(x, self.weights)
        w_times_x = torch.sum(w_times_x,dim=[2,5,6,7])
        w_times_x = torch.add(w_times_x, self.bias)

        return w_times_x


class Tree3(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_groups = 3
        self.num_filters_conv1 = 15
        self.num_filters_conv2 = 16
        self.activation = "sigmoid"

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.num_filters_conv1*self.num_groups, kernel_size=5, groups=self.num_groups)
        self.tree1 = TreeLayer((1,self.num_filters_conv2,self.num_filters_conv1,self.num_groups,7,7,2,2),self.activation)
        self.fc1 = nn.Linear(7*self.num_filters_conv2*3, 10 )

        # Initializations
        if self.activation == "relu":
            nn.init.kaiming_normal_(self.conv1.weight)
        elif self.activation == "sigmoid":
            nn.init.normal_(self.conv1.weight,mean=0.0,std=1.0)

        if self.activation == "relu":
            nn.init.kaiming_normal_(self.fc1.weight)
        elif self.activation == "sigmoid":
            nn.init.normal_(self.fc1.weight,mean=0.0,std=1.0)

    def forward(self, x):
        if self.activation == "relu":
            out = F.relu(self.conv1(x))
        elif self.activation == "sigmoid":
            out = torch.sigmoid(self.conv1(x))

        out = F.max_pool2d(out, 2)
        out = F.unfold(out,(2,2),stride=2)

        out = out.reshape(-1,1*self.num_filters_conv1*self.num_groups,2*2,7*7).transpose(2,3)
        out = out.reshape(-1,self.num_groups,1,self.num_filters_conv1,7,7,2,2).transpose(1,2).transpose(2,3)

        if self.activation == "relu":
            out = F.relu(self.tree1(out))
        elif self.activation == "sigmoid":
            out = torch.sigmoid(self.tree1(out))

        out = out.reshape(out.size(0),7*self.num_filters_conv2*3)
        out =self.fc1(out)

        return out
