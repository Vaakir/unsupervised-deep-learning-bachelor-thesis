import numpy as np
from skimage.metrics import structural_similarity as ssim

def NMSE(pred, test):
    return np.sum((pred.flatten() - test.flatten())**2) / np.sum(test**2)

def NRMSE(pred, test):
    """Normalized Root Mean Squared Error"""
    return np.sqrt(np.mean((pred.flatten() - test.flatten()) ** 2)) / (np.max(test) - np.min(test))

def SSIM(pred, test):
    """Structural Similarity Index (SSIM)"""
    if pred.shape != test.shape:
        pred.reshape(test.shape)
    return ssim(pred, test, data_range=test.max() - test.min())