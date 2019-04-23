# ********************* Import Libraries ********************************

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

# ******************************* Define the Model Class (Net) ****************************
class Net(nn.Module):

    def __init__(self):
        # This super method is called to call the __init__ class of the nn.Module class that this Net class inherits
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 5)
        #output_size = (224-5)/1 + 1 = 220 therefore, (16, 220, 220)
        
        self.pool1 = nn.MaxPool2d(2,2)
        #output_size = (16, 110, 110)
        
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 5)
        #output_size = (110-5)/1 + 1 = 106 therefore, (32,106,106)
        
        self.pool2 = nn.MaxPool2d(2,2)
        #output_size = (32,26,26)
        
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32,64,5)
        #output_size = (53-5)/1 + 1 = 52 therefore, (64, 49, 49)
        
        self.pool3 = nn.MaxPool2d(2,2)
        #output_size = (64,24,24)
        
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, 3)
        #output_size = (24-3)/1 + 1 = 220 therefore, (128, 22, 22)
        
        self.pool4 = nn.MaxPool2d(2,2)
        #output_size = (128, 11, 11)
        
        self.bn4 = nn.BatchNorm2d(128)
        
        #The following are the dense layers (fully connected)
        self.linear1 = nn.Linear(128*11*11, 1000)
        
        self.linear3 = nn.Linear(1000, 136)
       

        
    def forward(self, x):
        
        # Conv/Pool layer 1
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        
        # Conv/Pool layer 2
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        
        # Conv/Pool layer 3
        x = self.pool3(self.bn3(F.relu(self.conv3(x))))
        
        # Conv/Pool layer 4
        x = self.pool4(self.bn4(F.relu(self.conv4(x))))
        
        # Flatten for the Fully Connected Layers
        x = x.view(x.size(0), -1)
        
        # Densely Connected Layers
        x = F.relu(self.linear1(x))
        
        x = self.linear3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        # the output x is not ran through a softmax for a example because we are not looking for class probabilities here but raw numbers (x, y coords)
        return x
