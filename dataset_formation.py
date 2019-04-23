# ************************************* Import the required libraries ****************************************************
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
# PyTorch has its own Dataset and DataLoader classes
from torch.utils.data import Dataset, DataLoader



# ************************************* Create the Dataset Loader Class **************************************************

# Loading a dataset this way is Performance efficient because it doesnt load every image in memory at the same time
# Must do this every time!!!!!
class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""
    # This is the function that is called whenever you instantiate an instance of the FacialKeypointsDataset class
    # The arguments in this __init__ function are the parameters needed when you create the instance
    # For the transform, it is saying, if no transform is given, default to None
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # get dataframe of key points like we have seen before
        self.key_pts_frame = pd.read_csv(csv_file)
        # set the root directory of the images
        self.root_dir = root_dir
        # Can supply a transform for the data (keypoints & images) otherwise, default to no transform 
        self.transform = transform
        
    # This overrides the original len function (from python)
    # This makes it so len(dataset) calls this function instead of the normal shit
    def __len__(self):
        return len(self.key_pts_frame)
    # this overrides the get item function to make it return a 2 key dict of image matrix and 2d array of its keypoints
    # this makes it so this function gets called when you do something like "dataset[some idx]" instead of the normal functionality of "arrayLikeDataStructure[idx]"
    def __getitem__(self, idx):
        # the os.path.join function constructs a pathname out of one or more partial pathnames
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        # Read in the image
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        # turn the dataframe into a matrix and reshape to nx2
        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        # when you reshape with -1 as a paramater value, you are saying take the number that is provided already (aka n in this case)
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        # If there is a transform, apply it to the image
        if self.transform:
            sample = self.transform(sample)

        return sample


# ***************************************** Define Your Transform Functions *************************************************************

# you only need a __init__ function in a class if you plan on having parameters when you instantiate it, otherwise
# no need for it 


import torch
from torchvision import transforms, utils
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    # https://stackoverflow.com/questions/9663562/what-is-the-difference-between-init-and-call
    # the difference between __init__ and __call__:
    # class Foo:
    # def __init__(self, a, b, c):
    #      ...

    # x = Foo(1, 2, 3) # __init__

    # class Foo:
    #     def __call__(self, a, b, c):
    #          ...

    # x = Foo()
    # x(1, 2, 3) # __call__

    # Basically, __init__ is for when you instantiate class with arguments, __call__ is when you call the class on some arguments after instantiating

    def __call__(self, sample):
        # The sample is a dict of image and keypoints
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


# Tensors are similar to numpy ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing
# Tensors are multi dimensional Matrices
# This class can be called on a dict of image and keypoints and convert it into a tensor
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}


