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
    """SSIM loss for both 2D and 3D images"""

    # If the image is 2D (batch_size, height, width)
    if len(y_true.shape) == 3:  # (batch_size, height, width)
        y_true_copy = tf.expand_dims(y_true, axis=-1)  # Add a channel dimension
        y_pred_copy = tf.expand_dims(y_pred, axis=-1)  # Add a channel dimension
        return 1.0 - tf.reduce_mean(tf.image.ssim(y_true_copy, y_pred_copy, max_val=1.0))
    
    # If the image is 3D (batch_size, height, width, depth)
    elif len(y_true.shape) == 4:  # (batch_size, height, width, depth)
        # Compute SSIM for each slice along the depth axis (if depth exists in the last dimension)
        ssim_scores = []
        for i in range(y_true.shape[-1]):  # Loop over the depth dimension
            slice_true = y_true[:, :, :, i]  # Take the i-th slice from y_true
            slice_pred = y_pred[:, :, :, i]  # Take the i-th slice from y_pred
            slice_true = tf.expand_dims(slice_true, axis=-1) # add channel dimension
            slice_pred = tf.expand_dims(slice_pred, axis=-1) # add channel dimension
            #print(slice_true.shape, slice_pred.shape, y_true.shape)
            ssim_score = tf.image.ssim(slice_true, slice_pred, max_val=1.0)
            ssim_scores.append(ssim_score)

        # Average SSIM scores over all slices
        average_ssim = tf.reduce_mean(ssim_scores)
        return 1.0 - average_ssim
    else:
        raise ValueError("Input tensors must be either 3D (for 2D images) or 4D (for 3D images).")
