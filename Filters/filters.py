import numpy as np
import cv2

def edge_detect(im, k = 1, abs_ = False, alpha = 0.5):
    """ 
    :params:
    - im: The image(s) to perform edge detection on.
    - k: The kernel size, ie how far to roll before taking the difference in each axis.
    - abs_: Should we use the absolute or the signed difference?
    - alpha: How much of the original image bleeds through.
    :returns:
    - An image of the same size as input images.
    """
    edge = 3*im - np.roll(im, k, -1) - np.roll(im, k, -2) - np.roll(im, k, -3)
    if abs_: edge = np.abs(edge)
    return edge + (im-edge) * alpha

def preprocess_images(img_list, processes=[]):
    """
    Preprocess a list of images by applying a series of transformations.

    :Parameters:
    - img_list (list of ndarray): A list of images in BGR format.
    - processes (list of str): A list of process names to apply. Supported processes are:
        - "Resize32": Resize image to 32x32 pixels.
        - "Resize64": Resize image to 64x64 pixels.
        - "Resize128": Resize image to 128x128 pixels.
        - "Gaussian": Apply Gaussian blur with a 5x5 kernel.
        - "Normalize": Normalize pixel values to the range [0, 1].
        - "SobelX": Apply Sobel filter to detect horizontal edges.
        - "SobelY": Apply Sobel filter to detect vertical edges.
        - "Laplacian": Apply Laplacian filter for edge detection.
        - "Canny": Apply Canny edge detection.

    Returns:
    - list of ndarray: A list of processed images.
    """
    process_map = {
        "Resize32": lambda img: cv2.resize(img, (32, 32)),
        "Resize64": lambda img: cv2.resize(img, (64, 64)),
        "Resize128": lambda img: cv2.resize(img, (128, 128)),
        "Gaussian": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
        "Normalize": lambda img: img / 255.0,
        "SobelX": lambda img: cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3),
        "SobelY": lambda img: cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3),
        "Laplacian": lambda img: cv2.Laplacian(img, cv2.CV_64F),
        "Canny": lambda img: cv2.Canny(img, 50, 150),
    }

    processed_images = []
    
    for img in img_list:
        # Check if the image already has 3 channels, if not, skip conversion
        if len(img.shape) == 3 and img.shape[2] == 3:  # If image has 3 channels (BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply each process in the 'processes' list to the image
        for process_name in processes:
            if process_name in process_map:
                img = process_map[process_name](img)
        
        # Append the processed image to the list
        processed_images.append(img)

    return np.array(processed_images)