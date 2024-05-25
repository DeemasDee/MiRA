import torch
import torch.nn as nn
import torchvision.models as models

class LiverClassifier(nn.Module):
    def __init__(self):
        super(LiverClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        x = self.resnet(x)
        return x

# Usage
model = LiverClassifier()
print(model)
