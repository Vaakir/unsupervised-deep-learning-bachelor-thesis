import matplotlib.pyplot as plt 
import numpy as np
import nibabel as ni 
import torch

"""
def display_image(x, y):
    assert x.shape == y.shape
    if x.dim() == 5:
        x = torch.reshape(x, (x.shape[0], 80, 96, 80)).cpu().detach().numpy()
        y = torch.reshape(y, (y.shape[0], 80, 96, 80)).cpu().detach().numpy()
        for i in range(len(x)):
            rows = 2
            columns = 1
            fig=plt.figure()
            for idx in range(rows*columns):
                fig.add_subplot(rows, columns, idx+1)
                if idx < columns:
                    plt.imshow(x[i, :, 57, :], cmap="gray", origin="lower")
                else:
                    plt.imshow(y[i, :, 57, :], cmap="gray", origin="lower")
            plt.show()
"""
import matplotlib.pyplot as plt
import torch

def display_image2(x, y):
    assert x.shape == y.shape  # Ensure both inputs have the same shape
    if x.dim() == 5:
        # Flatten the 5D input to 4D (batch_size, channels, depth, height, width)
        x = x.squeeze(1)  # Remove the channel dimension (assuming single channel)
        y = y.squeeze(1)  # Same for the target
        
        # Loop through the batch
        for i in range(len(x)):
            rows = 3
            columns = 2
            fig = plt.figure()
            for idx in range(rows * columns):
                fig.add_subplot(rows, columns, idx + 1)
                if idx < columns:
                    # Display a slice from the middle of the depth dimension (assuming the middle slice)
                    plt.imshow(x[i, x.shape[1] // 2, :, :].T, cmap="gray", origin="lower")
                else:
                    plt.imshow(y[i, y.shape[1] // 2, :, :].T, cmap="gray", origin="lower")
                
            plt.show()


#import matplotlib.pyplot as plt
#import torch  # Needed if x, y are PyTorch tensors

def display_image(x, y):
    assert x.shape == y.shape, "Shapes of x and y must match!"
    
    if isinstance(x, torch.Tensor):
        x, y = x.cpu().detach().numpy(), y.cpu().detach().numpy()  # Convert to NumPy
    
    if x.ndim == 5:  # Shape: [batch, channels, depth, height, width]
        x = x.squeeze(1)  # Remove the channel dimension
        y = y.squeeze(1)

    diff = x - y  # Compute difference
    
    # Loop through the batch
    for i in range(len(x)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 3 columns: x, y, x - y
        
        middle_slice = x.shape[1] // 2  # Middle depth slice
        
        axes[0].imshow(x[i, middle_slice, :, :].T, cmap="gray", origin="lower")
        axes[0].set_title("Input (x)")
        
        axes[1].imshow(y[i, middle_slice, :, :].T, cmap="gray", origin="lower")
        axes[1].set_title("Target (y)")
        
        axes[2].imshow(diff[i, middle_slice, :, :].T, cmap="bwr", origin="lower")  # bwr highlights differences
        axes[2].set_title("Difference (x - y)")
        
        plt.show()
