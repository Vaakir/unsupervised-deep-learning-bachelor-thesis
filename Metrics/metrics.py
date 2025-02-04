import numpy as np
from skimage.metrics import structural_similarity as ssim

def MSE(pred, actual):
    return np.mean((pred.flatten() - actual.flatten())**2)

def NMSE(pred, actual):
    return np.sum((pred.flatten() - actual.flatten())**2) / np.sum(actual**2)

def NRMSE(pred, actual):
    """Normalized Root Mean Squared Error"""
    return np.sqrt(np.mean((pred.flatten() - actual.flatten()) ** 2)) / (np.max(actual) - np.min(actual))

def SSIM(pred, actual):
    """Structural Similarity Index (SSIM)"""
    if pred.shape != actual.shape:
        pred.reshape(actual.shape)
    return ssim(pred, actual, data_range=actual.max() - actual.min())