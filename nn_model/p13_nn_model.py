import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input +1
        return output

model = Model()
x = torch.tensor(1.0)
ouput = model(x)
print(ouput)