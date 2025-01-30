"""
Author: Duy-Phuong Dao
Email: phuongdd.1997@gmail.com (or duyphuongcri@gmail.com)
"""

import torch
import numpy as np
#import nibabel as ni
import nibabel as nib
import os, shutil
import time
import random
import pandas as pd


def split_train_test(dir, ratio_test=0.15):
    if not os.path.exists(os.path.join(dir, "train")): os.mkdir(os.path.join(dir, "train"))
    if not os.path.exists(os.path.join(dir, "test")): os.mkdir(os.path.join(dir, "test"))
    
    images_list = [i for i in os.listdir(dir) if i.endswith(".nii.gz")]

    random.shuffle(images_list)
    threshold = int(len(images_list)*ratio_test)
    train_list = images_list[:-threshold]
    test_list = images_list[-threshold:]

    for i in train_list:
        shutil.move(os.path.join(dir, i), os.path.join(dir, "train", i))
    for i in test_list:
        shutil.move(os.path.join(dir, i), os.path.join(dir, "test", i))

def save_data_to_csv(dir, z):
    pd.DataFrame(z).to_csv(dir, header=None, index=False)

"""
def load_mri_images(path, batch_size):
    filenames = [i for i in os.listdir(path) if i.endswith(".nii.gz")] #and i.startswith("norm_023_S_0030")
    random.shuffle(filenames) #, random.random)
    n = 0
    while n < len(filenames):
        batch_image = []
        for i in range(n, n + batch_size):
            if i >= len(filenames):
                ##n = i
                break
            #print(filenames[i])
            image = nib.load(os.path.join(path, filenames[i]))
            image = np.array(image.dataobj)
            image = np.pad(image, ((1,0), (1,0), (1, 0)), "constant", constant_values=0)
            image = torch.Tensor(image)
            image = torch.reshape(image, (1,1, 80, 96, 80))
            #image = (image - image.min()) / (image.max() - image.min())
            image = image / 255.
            batch_image.append(image)
        n += batch_size
        batch_image = torch.cat(batch_image, axis=0)
        yield batch_image
"""

def load_mri_images_old(path, batch_size):
    # Get list of .nii.gz files
    filenames = [i for i in os.listdir(path) if i.endswith(".nii.gz")]
    random.shuffle(filenames)
    n = 0

    while n < len(filenames):
        batch_image = []
        for i in range(n, n + batch_size):
            if i >= len(filenames):
                break
            # Load the image using nibabel
            image = nib.load(os.path.join(path, filenames[i]))
            image = image.get_fdata()  # Get the image data as a numpy array

            # Normalize image values to [0, 1]
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            
            # Ensure shape compatibility with the model
            image = torch.tensor(image, dtype=torch.float32)  # Convert to PyTorch tensor
            image = torch.unsqueeze(image, dim=0)  # Add channel dimension
            image = torch.unsqueeze(image, dim=0)  # Add batch dimension

            batch_image.append(image)

        n += batch_size
        if batch_image:  # Ensure non-empty batch
            batch_image = torch.cat(batch_image, dim=0)
            yield batch_image


def load_mri_images(path, batch_size):
    filenames = [i for i in os.listdir(path) if i.endswith(".nii.gz")] #and i.startswith("norm_023_S_0030")
    random.shuffle(filenames)
    n = 0
    while n < len(filenames):
        batch_image = []
        for i in range(n, n + batch_size):
            if i >= len(filenames):
                ##n = i
                break
            #print(filenames[i])
            image = nib.load(os.path.join(path, filenames[i]))
            #print("Original image shape:", image.shape, "vs (80,96,80)")

            image = np.array(image.dataobj)
            #image = np.pad(image, ((1,0), (1,0), (1, 0)), "constant", constant_values=0)
            image = torch.Tensor(image)
            
            # image = torch.reshape(image, (1, 1) + image.shape)  # Keeps original shape if unknown
            image = torch.reshape(image, (1,1, 208, 240, 256))
            #image = (image - image.min()) / (image.max() - image.min())
            image = image / 255.
            batch_image.append(image)
        n += batch_size
        batch_image = torch.cat(batch_image, axis=0)
        yield batch_image

#################### TEST #################   
path = "C:/Users/kiran/Documents/_UIS/sem6/BACH/Data/_testfew/" 
start = time.time()
loaded = load_mri_images(path, 2)
for i in loaded:
    print(time.time()-start)
    start = time.time()
    print(i.shape)


#split_train_test("/home/ubuntu/Desktop/DuyPhuong/VAE/data")