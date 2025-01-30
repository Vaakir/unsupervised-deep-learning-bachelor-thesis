import glob
from scipy.ndimage import zoom
import numpy as np
from tqdm import tqdm
import nibabel as nib

def open_images(path, crop=True, target_size = (80,96,80), normalize=True):
    files = glob.glob(path)

    images = []
    for file in tqdm(files[:250],"Loading images"):
        img = nib.load(file).get_fdata()

        if crop:
            brain_mask = img > 0.01
            bounds = np.where(brain_mask)
            x_min, x_max, y_min, y_max, z_min, z_max = np.min(bounds[0]), np.max(bounds[0]), np.min(bounds[1]), np.max(bounds[1]), np.min(bounds[2]), np.max(bounds[2])
            brain_img = img[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

        if target_size != None:
            zoom_factors = [t / b for t, b in zip(target_size, brain_img.shape)]
            brain_img = zoom(brain_img, zoom_factors, order=1)  # Linear interpolation

        if normalize:
            q2m = .785700/.475665
            fac = np.min([q2m / np.quantile(brain_img,0.98), 1. / np.max(brain_img)])
            brain_img *= fac

        images.append(brain_img)

    return np.stack(images)