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
import torch.nn.functional as F


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


def load_mri_images_newer(path, batch_size, downscale = 4):
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
            image = nib.load(os.path.join(path, filenames[i])) # Load image
            #print("Original image shape:", image.shape, "vs (80,96,80)")
            image = np.array(image.dataobj)
            # image = np.pad(image, ((1,0), (1,0), (1, 0)), "constant", constant_values=0)

            # Crop based on percentage from the end
            depth, height, width = image.shape
            x_min, x_max = int(width * 0.1), int(width * 0.9)    # z of displayed image
            y_min, y_max = int(height * 0.1), int(height * 0.82) # y of displayed image
            z_min, z_max = int(depth * 0.32), int(depth * 0.99)  # x of displayed image
            cropped_image = image[x_min:x_max, y_min:y_max, z_min:z_max] # Crop the image to the bounding box
            image = torch.Tensor(cropped_image) # Convert cropped image to tensor

            #image = torch.Tensor(image) # conver to tensor
            
            # image = torch.reshape(image, (1, 1) + image.shape)  # Keeps original shape if unknown
            # image = torch.reshape(image, (1,1, 208, 240, 256))
            #image = (image - image.min()) / (image.max() - image.min())

            image = image / 255.  # Normalize (assuming max pixel value is 255)
            
            # Ensure correct shape (1, C, D, H, W)
            image = image.unsqueeze(0).unsqueeze(0)  # Add batch & channel dimensions

            # Expected original shape (208, 240, 256)
            image = F.interpolate(image, scale_factor=(1/downscale, 1/downscale, 1/downscale), mode="trilinear", align_corners=False)
            

            # image = image / 255.
            batch_image.append(image)
        n += batch_size
        batch_image = torch.cat(batch_image, axis=0)
        yield batch_image

def get_mri_shape(path, downscale=1):
    "Returns the mri shape after cropping and downscaling"
    filenames = [f for f in os.listdir(path) if f.endswith(".nii.gz")]

    image = nib.load(os.path.join(path, filenames[0])).get_fdata()  # Load image as numpy array
    # Crop the image
    depth, height, width = image.shape
    x_min, x_max = int(width * 0.3), int(width * 0.78)    
    y_min, y_max = int(height * 0.14), int(height * 0.78)
    z_min, z_max = int(depth * 0.32), int(depth * 0.99)    
    image = image[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Convert to tensor and normalize
    image = torch.tensor(image, dtype=torch.float32) / 255.0  
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch & channel dimensions
    
    # Downscale
    image = F.interpolate(image, scale_factor=(1/downscale, 1/downscale, 1/downscale), 
                          mode="trilinear", align_corners=False)
    
    # Ensure dimensions are divisible by 16
    depth, height, width = image.shape[2], image.shape[3], image.shape[4]
    new_depth = ((depth + 15) // 16) * 16
    new_height = ((height + 15) // 16) * 16
    new_width = ((width + 15) // 16) * 16
    
    # Pad the image to make its dimensions divisible by 16
    pad_depth = new_depth - depth
    pad_height = new_height - height
    pad_width = new_width - width
    
    # Apply padding (padding is applied as [left, right, top, bottom, front, back])
    image = F.pad(image, (0, pad_width, 0, pad_height, 0, pad_depth))

    return image.shape


def load_mri_images(path, batch_size, downscale=1):
    """Loads MRI images in batches, preprocesses, and normalizes them."""
    filenames = [f for f in os.listdir(path) if f.endswith(".nii.gz")]
    random.shuffle(filenames)  # Shuffle dataset

    for i in range(0, len(filenames), batch_size):
        batch = []
        batch_filenames = filenames[i : i + batch_size]
        for file in batch_filenames:
            image = nib.load(os.path.join(path, file)).get_fdata()  # Load image as numpy array
            
            # Crop the image
            depth, height, width = image.shape
            x_min, x_max = int(width * 0.3), int(width * 0.78)    
            y_min, y_max = int(height * 0.14), int(height * 0.78)
            z_min, z_max = int(depth * 0.32), int(depth * 0.99)  
            image = image[z_min:z_max, y_min:y_max, x_min:x_max]
            
            # Convert to tensor and normalize
            image = torch.tensor(image, dtype=torch.float32) / 255.0  
            image = image.unsqueeze(0).unsqueeze(0)  # Add batch & channel dimensions
            
            # Downscale
            image = F.interpolate(image, scale_factor=(1/downscale, 1/downscale, 1/downscale), 
                                  mode="trilinear", align_corners=False)
            
            
            # Ensure dimensions are divisible by 16
            depth, height, width = image.shape[2], image.shape[3], image.shape[4]
            new_depth = ((depth + 15) // 16) * 16
            new_height = ((height + 15) // 16) * 16
            new_width = ((width + 15) // 16) * 16
            
            # Pad the image to make its dimensions divisible by 16
            pad_depth = new_depth - depth
            pad_height = new_height - height
            pad_width = new_width - width

            image = F.pad(image, (0, pad_width, 0, pad_height, 0, pad_depth))

            batch.append(image)

        yield torch.cat(batch, dim=0)  # Stack tensors to form a batch


if __name__ == "__main__":
    #################### TEST #################   
    path = "C:/Users/kiran/Documents/_UIS/sem6/BACH/Data/_testfew/"
    start = time.time()
    loaded = load_mri_images(path, 2)
    for i in loaded:
        print(time.time()-start)
        start = time.time()
        print(i.shape)


    #split_train_test("/home/ubuntu/Desktop/DuyPhuong/VAE/data")
    import matplotlib.pyplot as plt
    def plot_mri(image_tensor):
        "Plots a middle slice of a 3D MRI scan."
        image_np = image_tensor.squeeze().numpy()  # Remove batch & channel dimensions
        middle_slice = int(image_np.shape[0] * 0.5)  # Get the middle slice
        plt.imshow(image_np[middle_slice, :, :], cmap="gray")  # Plot axial view
        plt.title("Middle Slice of MRI")
        plt.axis("off")
        plt.show()

    # Load images
    path = "C:/Users/kiran/Documents/_UIS/sem6/BACH/Data/_testfew/" 
    loaded_images = load_mri_images(path, batch_size=2, downscale=3)

    # Get first batch and plot
    for batch in loaded_images:
        print("Batch shape:", batch.shape)
        plot_mri(batch[0])  # Plot the first image in batch
        break  # Stop after plotting one batch
