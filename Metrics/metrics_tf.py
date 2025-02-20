import tensorflow as tf

# METRICS
def MSE_loss(y_true, y_pred):
    """Mean Squared Error (MSE) loss"""
    return tf.reduce_mean(tf.square(y_true - y_pred))

def NMSE_loss(y_true, y_pred):
    """Normalized Mean Squared Error (NMSE) loss"""
    return tf.reduce_sum(tf.square(y_true - y_pred)) / tf.reduce_sum(tf.square(y_true))

def NRMSE_loss(y_true, y_pred):
    """Normalized Root Mean Squared Error (NRMSE) loss"""
    range_val = tf.reduce_max(y_true) - tf.reduce_min(y_true)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred))) / (range_val + 1e-8)  # Avoid division by zero

def SSIM_loss(y_true, y_pred):
    """SSIM loss (1 - SSIM for minimization)"""
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
