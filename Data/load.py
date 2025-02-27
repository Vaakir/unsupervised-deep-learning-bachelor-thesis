from glob import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom
import os
from tensorflow.keras import utils

def load(
        dataset_name:str = "Pre-processed",
        crop:bool = True,
        normalize:bool = True,
        #subvoxel_rolling_augmentation:bool = True,
        target_size:tuple|None = (80,96,80),
        train_test_split:float = 0.9,
        take:int=-1,
        subdirs=[]
        ):
    """
    :params:
    - dataset_name: Path relative to this script to the parent folder of the dataset.
    - crop: If True: images are cut in all axes s.t. only the non-zero voxels are included.
    - normalize: If True: magnitude of voxels are normalized between 0 and 1.
    - target_size: Downsampling target size. Set to None to resample to lowest cropped size.
    - train_test_split: The ratio of images used in the train set. Set to 1 to include all loaded images in a single dataset.
    - take: How many images to load. Set to -1 to load all the images in the dataset.
    """
    #- subvoxel_rolling_augmentation: If True: rolls the images around some times, producing a richer dataset.
    if len(subdirs) > 0:
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for i, group in enumerate(subdirs):
            data = load(
                f"{dataset_name}/{group}",
                crop,
                normalize,
                target_size,
                train_test_split,
                take
                )
            if len(data) == 2:
                x_train.extend(data[0])
                x_test.extend(data[1])
                y_train.extend([i]*len(data[0]))
                y_test.extend([i]*len(data[1]))
            else:
                x_train.extend(data)
                y_train.extend([i]*len(data))

        y_train = utils.to_categorical(y_train)
        y_test = utils.to_categorical(y_test)
        x_train = np.stack(x_train)
        x_test = np.stack(x_test)
        return x_train, y_train, x_test, y_test
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
    dataset_path = os.path.join(script_dir, dataset_name)  # Create the absolute path
    files = glob(os.path.join(dataset_path, "*.nii.gz"))  # Use absolute path for glob
    if take >= 0:
        files = files[:take]

    images = []

    def handle_image(img):
        if crop:
            brain_mask = img > 0.01
            bounds = np.where(brain_mask)
            x_min, x_max, y_min, y_max, z_min, z_max = np.min(bounds[0]), np.max(bounds[0]), np.min(bounds[1]), np.max(bounds[1]), np.min(bounds[2]), np.max(bounds[2])
            img = img[x_min-2:x_max+3, y_min-2:y_max+3, z_min-2:z_max+3]

        '''
        if subvoxel_rolling_augmentation:
            # Since we zoom the image, we can nudge the picture in various directions to
            # achieve new sub-pixel level information in the output :)
            for roll in range(2):
                for axis in range(3):
                    loaded_images.append(np.roll(img, roll*2, axis))
        '''
                    
        if normalize:
            q2m = .785700/.475665
            fac = np.min([q2m / np.quantile(img,0.98), 1. / np.max(img)])
            img *= fac
            
        if target_size != None:
            zoom_factors = [t / b for t, b in zip(target_size, img.shape)]
            img = zoom(img, zoom_factors, order=1)  # Linear interpolation


        return [img]
    
    desc = f"Loading {dataset_name.split('/')[-1]}"
    for file in tqdm(files, desc):
        img = nib.load(file).get_fdata()
        images.extend(handle_image(img))

    images = np.stack(images)
    if train_test_split>=1 or train_test_split<=0: return images

    idx_split = int(len(images) * train_test_split)
    train, test = images[:idx_split], images[idx_split:]

    return train, test


def load_middle_slices(
        dataset_name: str = "Pre-processed",  # Now takes the full dataset path instead of constructing it
        axis=["sagittal","axial","coronal"][0],
        crop: bool = True,
        normalize: bool = True,
        target_size: tuple | None = (80, 96),  # Only height and width since we extract a 2D slice
        train_test_split: float = 0.9,
        take: int = -1
    ):
    """
    :params:
    - dataset_path: Full path to the dataset folder.
    - crop: If True, images are cut in all axes such that only the non-zero voxels are included.
    - normalize: If True, voxel magnitudes are normalized between 0 and 1.
    - target_size: Downsampling target size. Set to None to keep original size.
    - train_test_split: Ratio of images used in the train set. Set to 1 to load everything into a single dataset.
    - take: Number of images to load. Set to -1 to load all available images.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
    dataset_path = os.path.join(script_dir, dataset_name)  # Create the absolute path
    files = glob(os.path.join(dataset_path, "*.nii.gz"))  # Use absolute path for glob
    if take >= 0:
        files = files[:take]

    images = []

    def handle_image(img):
        if crop:
            brain_mask = img > 0.01
            bounds = np.where(brain_mask)
            x_min, x_max, y_min, y_max, z_min, z_max = (
                np.min(bounds[0]), np.max(bounds[0]),
                np.min(bounds[1]), np.max(bounds[1]),
                np.min(bounds[2]), np.max(bounds[2])
            )
            img = img[x_min-2:x_max+3, y_min-2:y_max+3, z_min-2:z_max+3]

        # Extract middle slice (x-axis)
        axes = {"sagittal": 0, "coronal": 1, "axial": 2}
        middle_x = img.shape[axes[axis]] // 2
        img = img.take(middle_x, axis=axes[axis])

        if normalize:
            q2m = .785700 / .475665
            fac = np.min([q2m / np.quantile(img, 0.98), 1. / np.max(img)])
            img *= fac

        if target_size is not None:
            zoom_factors = [t / b for t, b in zip(target_size, img.shape)]
            img = zoom(img, zoom_factors, order=1)  # Linear interpolation

        return [img]

    for file in tqdm(files, "Loading images"):
        img = nib.load(file).get_fdata()
        images.extend(handle_image(img))

    images = np.stack(images)

    if train_test_split >= 1 or train_test_split <= 0:
        return images

    idx_split = int(len(images) * train_test_split)
    train, test = images[:idx_split], images[idx_split:]

    return train, test