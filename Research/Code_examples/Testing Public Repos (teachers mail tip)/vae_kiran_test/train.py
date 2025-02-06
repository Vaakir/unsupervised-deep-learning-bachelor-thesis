"""
Author: Duy-Phuong Dao
Email: phuongdd.1997@gmail.com (or duyphuongcri@gmail.com)
"""

import nibabel as ni
import numpy as np
import os, glob
import torch 
import csv
from tqdm import tqdm

import model
import loss
import dataloader
from dataloader import *

##---------Settings--------------------------
batch_size = 8
lrate = 0.01
epochs = 1000
weight_decay = 5e-7
##############
# C:\Users\kiran\Documents\_UIS\sem6\BACH\Data\_testfew
path_data = "C:/Users/kiran/Documents/_UIS/sem6/BACH/Data/208" # very_spatial_norm
path2save = "C:/Users/kiran/Documents/_UIS/sem6/BACH/Data/_test/test{}.pt"
dir_info = './infor'
os.makedirs(dir_info, exist_ok=True)  # Create the directory if it doesn't exist
f = open(os.path.join(dir_info,'model_vae_t1.csv'),'w',newline='')


####################
verbose = True
log = print if verbose else lambda *x, **i: None
np.random.seed(10)
torch.manual_seed(10)
###################
criterion_rec = loss.L1Loss()
criterion_dis = loss.KLDivergence()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(" GPU is activated" if device else " CPU is activated")
no_images = len(glob.glob(path_data + "/*.nii.gz"))
print("Number of MRI images: ", no_images)


# Load example image compressed:
# Send example image.shape to model.VAE constructor
# Adjust parameters in vae dynamically

downscale = 3
input_image_shape = get_mri_shape(path_data, downscale=downscale)


if __name__=="__main__":
    vae_model = model.VAE(latent_dim=1024, shape=input_image_shape)
    vae_model.to(device)
    #log(vae_model)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=lrate, weight_decay=weight_decay)
    # beta = 0.1  # Lower value to reduce the effect of KL divergence

    for epoch in tqdm(range(epochs)):
        loss_rec_batch, loss_KL_batch, total_loss_batch = 0, 0, 0
        loss_rec_epoch, loss_KL_epoch, total_loss_epoch = 0, 0, 0
        
        # training phrase
        vae_model.train()
        for batch_images in tqdm(dataloader.load_mri_images(path_data, batch_size, downscale=downscale)):
            optimizer.zero_grad()
            batch_images = batch_images.to(device)
            y, z_mean, z_log_sigma = vae_model(batch_images)
            
            # Measure loss
            loss_rec_batch = criterion_rec(batch_images, y)
            loss_KL_batch = criterion_dis(z_mean, z_log_sigma)

            # beta = min(1.0, epoch / 100.0)

            total_loss_batch = loss_rec_batch + loss_KL_batch

            # Optimize
            total_loss_batch.backward()
            optimizer.step()
            
            loss_rec_epoch += loss_rec_batch.item() * batch_images.shape[0]
            loss_KL_epoch += loss_KL_batch.item() * batch_images.shape[0]
            total_loss_epoch += total_loss_batch.item() * batch_images.shape[0]

        # save model
        log_info = (epoch + 1, epochs, loss_rec_epoch/no_images, loss_KL_epoch/no_images)
        log('%d/%d  Reconstruction Loss %.3f| KL Loss %.3f'% log_info)
        torch.save(vae_model, path2save.format(epoch+1))  

        # write csv
        writer = csv.writer(f)
        writer.writerow([epoch + 1, '{:04f}'.format(loss_rec_epoch/no_images), 
                                        '{:04f}'.format(loss_KL_epoch/no_images)])
    f.close()
