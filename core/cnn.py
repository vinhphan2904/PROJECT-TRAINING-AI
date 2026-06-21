from lib import *
from mapping import disease

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.set = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,len(disease))
        self.dropout = nn.Dropout(0.3)
    def forward(self,x):
        x = self.set(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



