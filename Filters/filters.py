import numpy as np

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
    return edge + (alpha-edge) * im