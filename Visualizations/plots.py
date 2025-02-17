import os
import math
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tensorflow as tf

# USEFULL FUNCTIONS
def plot_middle_slices_in_range(folder_path = r"C:\Users\kiran\Documents\_UIS\sem6\BACH\DementiaMRI\Data\Pre-processed", n1=0, n2=9, axis=1):
    "Function to plot the middle slices of images between n1 and n2 in a single figure"
    # Get a sorted list of all files in the directory excluding mask files
    all_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.nii.gz') and not f.endswith('_mask.nii.gz')]
    )
    
    # Filter files between n1 and n2
    selected_files = all_files[n1:n2]
    num_images = len(selected_files)
    
    # Determine grid size for the plot
    cols = math.ceil(math.sqrt(num_images))  # Number of columns in the grid
    rows = math.ceil(num_images / cols)  # Number of rows in the grid
    
    # Create a figure for plotting
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()  # Flatten to easily iterate over axes
    
    for i, file in enumerate(selected_files):
        file_path = os.path.join(folder_path, file)
        
        # Load the NIfTI file
        img = nib.load(file_path)
        img_data = img.get_fdata()
        
        # Calculate the middle index along axis 0
        middle_index = img_data.shape[axis] // 2
        
        # Plot the middle slice on the current axis
        axes[i].imshow(img_data[middle_index, :, :].T, cmap='gray', origin="lower")
        axes[i].set_title(file_path.split("\\")[-1].split("_")[0], fontsize=8)
        axes[i].axis('off')
    
    # Hide unused axes if the grid is larger than the number of images
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_models_training_time(models_list, fig_size=""):
    training_times = []
    names = []
    for m in models_list:
        if hasattr(m, 'training_time') and m.training_time is not None:
            training_times.append(m.training_time)
            names.append(m.name)

    if training_times:
        if fig_size: 
            plt.figure(figsize=fig_size)
        plt.bar(names, training_times)
        plt.xlabel('Model index')
        plt.ylabel('Training time (seconds)')
        plt.title('Training time of models')
        plt.xticks(rotation=45, ha='right')  # Rotate labels 45 degrees, align right
        plt.show()

def compare_models_loss_history(models, log10=True, fig_size="", title="Model's val_loss history"):
    """
    Compares the loss histories of multiple models by plotting their log-transformed losses.

    Parameters:
    - models: list of model objects, each containing a 'log.history["loss"]' attribute and a 'name'.

    Returns:
    - None: Displays a plot of the loss histories.
    """
    if fig_size: 
        plt.figure(figsize=fig_size)
    for i, model in enumerate(models):
        #color = decimal_to_rgb(i, len(models))
        #color = "#{:02x}{:02x}{:02x}".format(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        #y = [model.history.history["loss"], np.log10(model.history.history["loss"])][log10]
        
        # Sorry for the mess, but I wanted this to work for two different classes
        if isinstance(model.history, tf.keras.callbacks.History):
            y = [model.history.history["val_loss"], np.log10(model.history.history["val_loss"])][log10]
        else:
            y = [model.history["val_loss"], np.log10(model.history["val_loss"])][log10]
        plt.plot(y, label=model.name) #, color=color
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss Log10") if log10 else plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

def plot_images(images, dimensions, fig_size=(10,8), titles=[], top_title="Image Gallery", cmap="viridis"):
    """
    Plots a grid of images with corresponding titles.

    Parameters:
    - images: list of numpy arrays, the images to be displayed.
    - dimensions: tuple of int (rows, cols), grid layout for images.
    - fig_size: tuple for the figure size, eg (10,8)
    - titles: list of str, the titles for each image.
    - top_title: str, optional, the title of the overall figure.

    Returns:
    - None: Displays the plot.
    """
    plt.figure(figsize=fig_size)
    plt.suptitle(top_title, fontsize=12, fontweight="bold", y=0.99)
    for i, img in enumerate(images):
        plt.subplot(dimensions[0], dimensions[1], i + 1)
        plt.imshow(img, cmap=cmap)
        if len(titles) == len(images):
            plt.title(titles[i], fontsize=10)
        plt.axis('off')
        plt.tight_layout()
    plt.show()

def compare_models_reconstruction(brain_scan_id, models_list, test, loss):
    images = []
    titles = []
    for m in models_list:
        if m.VAE_model:
            latent = m.encode(test)[2]
        else:            
            latent = m.encode(test)
        recon = m.decode(latent)
        input_image = np.rot90(test[brain_scan_id][40].reshape((96, 80)))
        recon_image = np.rot90(recon[brain_scan_id][40])[:,:,0] # recon shape is 3dim we need -> 2d
        diff_image = input_image - recon_image
        images.extend([input_image, recon_image, diff_image])
        titles.extend([
            "Input", "Reconstructed", f"Difference ({round(loss(input_image, recon_image), 3)})"
        ])
    
    return images, latent, titles