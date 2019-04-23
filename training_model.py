# ************************** Necessary Imports ***************************************
import matplotlib.pyplot as plt
import numpy as np

# These are functions and libraries necessary to run the network you are bringing in
import torch
import torch.nn as nn
import torch.nn.functional as F

# Once you've define the network, you can instantiate it
# This is importing the Net Class from the models.py file within the same directory
from model import Net 

# Instantiating an instance of the Net Class (defined in the models.py file)
net = Net()


#***************************** Define Transforms and Training Dataset Instance ********************

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset class from dataset_formation.py
from dataset_formation import FacialKeypointsDataset

# the transforms we defined in dataset_formation.py 
from dataset_formation import Rescale, RandomCrop, Normalize, ToTensor


# order matters! i.e. rescaling should come before a smaller crop
# Rescale image to 250x250 --> random crop of 224x224 --> normalize pixel values --> convert to tensor
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

# create an instance of the FacialKeypointsDataset class
transformed_dataset = FacialKeypointsDataset(csv_file='/data/training_frames_keypoints.csv',
                                             root_dir='/data/training/',
                                             transform=data_transform)


# ******************* Batching and Loading Training Data ************************************

# load training data in batches (pre defined batch size, recommend to play around with this number)
batch_size = 10

# The DataLoader class from PyTorch gives you the configure how to batch/load from datasets
train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)


# ********************** Test Dataset Loading ******************************************

# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='/data/test_frames_keypoints.csv',
                                             root_dir='/data/test/',
                                             transform=data_transform)

# load test data in batches
batch_size = 10

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)

# ******************** Model Testing Function ***************************************

# test the model on a batch of test images

def net_sample_output():
    
    # iterate through the test dataset
    # The enumerate function provides an index which is great :)
    for i, sample in enumerate(test_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert image from regular tensor to FloatTensors
        images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)
        
        # reshape to batch_size x 68 x 2 pts
        # .view is like reshape but it only creates a view of it (aka doesnt change the original)
        # but, any changes you make on the original tensor will be reflected in the view
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        
        # break after first image is tested (again, only a test function)
        if i == 0:
            return images, output_pts, key_pts


# ********************** Visualizing output of CNN model *****************************

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):

    for i in range(batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the image data
        image = test_images[i].data   # get the image from it's Variable wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*50.0+100
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*50.0+100
        
        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
            
        plt.axis('off')

    plt.show()
    
# ********************************* Train Model ***************************************

import torch.optim as optim

# The algorithm you want to measure the error of the model
# It will be different depending on the type of response (SmoothL1Loss measures MAE, mean absolute error)
criterion = nn.SmoothL1Loss()

# This is the algorithm that updates via back propogation (Adam is a popular optimizer)
optimizer = optim.Adam(net.parameters())


def train_net(n_epochs):

    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights via backpropogation
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0

    print('Finished Training')



# ********************** Save Trained Model **********************************
## change the name to something uniqe for each new model
model_dir = 'saved_models/'
model_name = 'keypoints_model_1.pt'

# after training, save your model parameters in the dir 'saved_models'
torch.save(net.state_dict(), model_dir+model_name)