# from ...DementiaMRI.Architectures.VAE import VAE
#from Data.load import load
#from Architectures.latent_space_projections import pPCA, pTSNE, pUMAP, pISOMAP, pENCODED, plot_multiple_datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from Metrics.metrics import NMSE, SSIM, NRMSE, MSE
from Metrics.metrics_tf import MSE_loss, NMSE_loss, NRMSE_loss, SSIM_loss

import glob

import tensorflow as tf
from tensorflow.keras import layers, Model, models, Input, activations, regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from skimage.metrics import structural_similarity as ssim

import time
import math
import nibabel as nib
import json

# TODO: Add skip connections

# wasnt able to save vae's without this as a seperate class/function ig, ty jon :) (will implement when I have time, bruh, writing be slow ):
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, logvar = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * logvar) * epsilon

class AE(Model):
    def __init__(self, 
                input_shape, 
                latent_dim, 
                encoder_layers, 
                decoder_layers, 
                name="AE", 
                loss="mse", #of any other funciton
                learning_rate=1e-3,
                VAE_model=False,
                debug=False,
                GPU = False):
        super(AE, self).__init__(name=name)
        self.loss = loss
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.VAE_model = VAE_model
        self.history = None
        self.debug = debug
        self.GPU = GPU
        self.compile_model()
    
    def compile_model(self):
        """
        Compile the model by iterating through the provided layers and adding them to the Functional API model.
        """
        # Ensure TensorFlow is using the GPU
        #physical_devices = tf.config.list_physical_devices('GPU')
        for device in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        # if len(physical_devices) > 0:
            #tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        if len(self.input_shape) == 4: 
            self.input_shape = (list(self.input_shape) + [1])[-4:] # (8, 80, 96, 80) -> [8, 80, 96, 80, 1] -> [80, 96, 80, 1] 
        if len(self.input_shape) == 3:
            self.input_shape = (list(self.input_shape) + [1])[-3:] # (8, 80, 96) -> [8, 80, 96, 1] -> [80, 96, 1] 
        print(f"Interpreted image shape: {tuple(self.input_shape)}", end=" ")

        # 1. Build the encoder model
        input_layer = x = Input(shape=self.input_shape, name="input")
        x = self.add_layers(x, self.encoder_layers)

        # - Flatten the latent space (encoding output)
        print(f"Pre-flattened latent shape: {x.shape}")
        pre_flatten_shape = x.shape
        x = layers.Flatten()(x)

        # VAE specific: Output mean and log variance
        if self.VAE_model:
            self.mean = layers.Dense(self.latent_dim, name="mean")(x)
            self.log_var = layers.Dense(self.latent_dim, name="log_var")(x)
            z = layers.Lambda(self.sampling, name="z")([self.mean, self.log_var])
            self.encoder = Model(input_layer, [self.mean, self.log_var, z], name="encoder")
        else:
            x = layers.Dense(self.latent_dim, activation="leaky_relu")(x)
            self.encoder = Model(input_layer, x, name="encoder")

        # 2. Build the decoder model
        latent_input = x = layers.Input(shape=(self.latent_dim,)) 

        # Reshape the latent vector back to the original spatial dimensions
        x = layers.Dense(np.prod(pre_flatten_shape[1:]), activation="leaky_relu")(latent_input)
        x = layers.Reshape(pre_flatten_shape[1:])(x)
        x = self.add_layers(x, self.decoder_layers)
        
        self.decoder = Model(latent_input, x, name="decoder")

        # 3. Combine the encoder and decoder into a full VAE
        if self.VAE_model:
            mean, log_var, z = self.encoder(input_layer)
            autoencoder_output = self.decoder(z)
        else:
            autoencoder_output = self.decoder(self.encoder(input_layer))

        # 3. Combine the encoder and decoder into a full autoencoder
        self.autoencoder = Model(input_layer, autoencoder_output, name="autoencoder")
        
        # 4. GPU or CPU assignment (based on self.GPU)
        optimizer = Adam(learning_rate=self.learning_rate)
        loss_fn = self.vae_loss if self.VAE_model else self.loss
        
        with tf.device('/GPU:0' if self.GPU and tf.config.list_physical_devices('GPU') else '/CPU:0'):
            self.autoencoder.compile(optimizer=optimizer, loss=loss_fn)

    def add_layers(self, x, layers):
        for layer in layers:
            
            if isinstance(layer, tuple):
                layer_type, *layer_args = layer
                layer_kwargs = layer_args.pop() if isinstance(layer_args[-1], dict) else {}
                x = layer_type(*layer_args, **layer_kwargs)(x)
            else:
                x = layer(x)
        
            if self.debug:
                print(f"After {x.shape}: {layer}")
        return x

    def summary(self):
        """Print a summary of the neural network model."""
        if self.autoencoder is None:
            raise ValueError("The model is not compiled yet. Call `compile_model()` first.")
        self.autoencoder.summary()
    
    def train(self, x_train, x_test, epochs=2, batch_size=16, verbose=False, save_path="", save_interval=100):
        start_time = time.time()

        # No reason to delete the loss history, if we stopped the sim at epoch10,
        # then we can continue it at another point
        if not hasattr(self, "history") or not self.history:
            #self.history = tf.keras.callbacks.History()  # Ensure history is initialized
            #self.history.history = {"loss": [], "val_loss": []}
            self.history = {"loss": [], "val_loss": []}

        if self.GPU and tf.config.experimental.list_physical_devices('GPU'):
            print(f"Training {self.name} on the GPU", end="")
            with tf.device('/GPU:0'):
                for epoch in range(1, epochs + 1):
                    history = self.autoencoder.fit(
                        x_train, x_train,
                        epochs=1,
                        batch_size=batch_size,
                        shuffle=True,
                        #validation_split=0.2,
                        validation_data=(x_test, x_test),
                        verbose=verbose
                    )
                    # self.history.history["loss"].extend(history.history["loss"])
                    # self.history.history["val_loss"].extend(history.history["val_loss"])
                    self.history["loss"].extend(history.history["loss"])
                    self.history["val_loss"].extend(history.history["val_loss"])

                    if save_path and epoch % save_interval == 0 and not self.VAE_model:
                        self.save(save_path, name=f"{self.name}")
        else:
            print(f"Training {self.name} on the CPU", end="")
            print(type(self.autoencoder), x_train.shape)
            for epoch in range(1, epochs + 1):
                history = self.autoencoder.fit(
                        x_train, x_train,
                        epochs=1,
                        batch_size=batch_size,
                        shuffle=True,
                        #validation_split=0.2,
                        validation_data=(x_test, x_test),
                        verbose=verbose
                )
                #self.history.history["loss"].extend(history.history["loss"])
                #self.history.history["val_loss"].extend(history.history["val_loss"])
                self.history["loss"].extend(history.history["loss"])
                self.history["val_loss"].extend(history.history["val_loss"])
                if save_path and epoch % save_interval == 0 and not self.VAE_model:
                    print(self.name)
                    self.save(save_path, name=f"{self.name}")
        
        self.training_time = time.time() - start_time
        print(f" - {round(self.training_time, 2)}s")
        return self.history

    def vae_loss_old(self, y_true, y_pred):
        """Compute the VAE loss (reconstruction loss + KL divergence loss)."""
        reconstruction_loss = self.loss(y_true, y_pred)

        # KL divergence loss
        # mean, log_var = self.encoder.input
        kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mean) - K.exp(self.log_var), axis=-1)

        # kl_loss = -0.5 * K.mean(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)

        # Total loss
        return K.mean(reconstruction_loss + kl_loss)
    
    def vae_loss(self, y_true, y_pred):
        """Compute the VAE loss (reconstruction loss + KL divergence loss)."""

        # Reconstruction loss (SSIM loss)
        reconstruction_loss = self.loss(y_true, y_pred)

        # Extract mean and log variance from the encoder's output
        mean, log_var, _ = self.encoder(y_true)  # Use actual values, not placeholders

        # KL divergence loss
        kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)

        # Total loss
        return K.mean(reconstruction_loss + kl_loss)

    def sampling(self, args):
        mean, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
        return mean + K.exp(0.5 * log_var) * epsilon
    
    def sample(self, mean, log_var):
        """Sample from the latent space distribution using the reparameterization trick."""
        # Reparameterization trick
        epsilon = tf.random.normal(shape=tf.shape(mean))  # Sample epsilon from N(0, 1)
        z = mean + tf.exp(0.5 * log_var) * epsilon  # Reparameterization
        return z

    def plot_evaluation(self):
        """Plot the training and validation accuracy/loss over epochs using training logs."""
        pd.DataFrame(self.history).plot(figsize=(8, 5))
        plt.ylabel("")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.show()

    def encode(self, x):
        """Encode input data x into its latent space representation."""
        return self.encoder.predict(x)

    def decode(self, y):
        """Decode latent space representation y into the original data space."""
        return self.decoder.predict(y)

    def save(self, models_path, name="AE"):
        """Save the autoencoder, encoder, and decoder models to disk, and additional initialization data."""
        # Define the folder path where models will be saved
        model_folder = os.path.join(models_path, f"{name}")
        epoch_folder = os.path.join(model_folder, "epoch_"+str(len(self.history["loss"])))
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(epoch_folder, exist_ok=True)
        
        # Save the models to the folder
        self.autoencoder.save(os.path.join(epoch_folder, f"{name}_autoencoder.keras"))
        self.encoder.save(os.path.join(epoch_folder, f"{name}_encoder.keras"))
        self.decoder.save(os.path.join(epoch_folder, f"{name}_decoder.keras"))
        print(f"Models saved to {epoch_folder} as '{name}_autoencoder.keras', '{name}_encoder.keras', '{name}_decoder.keras'.")

        # Save loss history if available
        if self.history is not None:
            history_path = os.path.join(epoch_folder, f"{name}_loss_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.history, f)
            print(f"Loss history saved to {history_path}")

        # Convert the layers and other details to strings
        try:
            print(name)
        except:
            print("\n\nname not found (!!!) \n\n")

        model_details = {
            "input_shape": str(self.input_shape),
            "latent_dim": str(self.latent_dim),
            "encoder_layers": str(self.encoder_layers),
            "decoder_layers": str(self.decoder_layers),
            "name": str(name),
            "loss": str(self.loss.__name__),
            "learning_rate": str(self.learning_rate),
            "VAE_model": str(self.VAE_model),
            "debug": str(self.debug),
            "GPU": str(self.GPU)
        }

        # Save the model details (converted to strings) in a JSON file
        details_path = os.path.join(epoch_folder, f"{name}_model_details.json")
        with open(details_path, 'w') as f:
            json.dump(model_details, f, indent=4)
        print(f"Model details saved to {details_path}")


    @staticmethod
    def open(autoencoder_path):
        """Load the autoencoder, encoder, and decoder models from disk using a single path."""
        if not os.path.exists(autoencoder_path):
            raise FileNotFoundError(f"Could not find autoencoder model at {autoencoder_path}.")

        path, autoencoder_filename = os.path.split(autoencoder_path)
        name = autoencoder_filename.replace("_autoencoder.keras", "")

        encoder_path = os.path.join(path, f"{name}_encoder.keras")
        decoder_path = os.path.join(path, f"{name}_decoder.keras")
        history_path = os.path.join(path, f"{name}_loss_history.json")
        details_path = os.path.join(path, f"{name}_model_details.json")

        if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
            raise FileNotFoundError(f"Could not find encoder or decoder models in {path}.")
        
        autoencoder = tf.keras.models.load_model(autoencoder_path, custom_objects={"MSE_loss": MSE_loss})
        encoder = tf.keras.models.load_model(encoder_path)
        decoder = tf.keras.models.load_model(decoder_path)

        # Create an instance of AE without initializing
        ae = AE.__new__(AE)
        ae.autoencoder = autoencoder
        ae.encoder = encoder
        ae.decoder = decoder

        # Set required attributes
        ae.name = name
        ae.built = True

        # Compile the loaded model manually
        ae.autoencoder.compile(optimizer='adam')

        print("Models loaded and compiled successfully.")

        with open(history_path) as f:
            ae.history = json.load(f) #ae.history = {'loss': [], 'val_loss': []}  



        with open(details_path) as f:
            ae.details = json.load(f)
            
            # Ensure these are boolean values, convert them if necessary # [False, True]["False" == "True"]
            ae.input_shape = eval(ae.details["input_shape"])
            ae.latent_dim = eval(ae.details["latent_dim"])
            ae.encoder_layers = ae.details["encoder_layers"]
            ae.decoder_layers = ae.details["decoder_layers"]
            ae.name = ae.details["name"]
            ae.loss = {"MSE_loss": MSE_loss, "SSIM_loss": SSIM_loss}[ae.details["loss"]]
            # ae.loss = ae.details["loss"]
            ae.learning_rate = ae.details["learning_rate"]
            ae.VAE_model = eval(ae.details["VAE_model"])
            ae.debug = eval(ae.details["debug"])
            ae.GPU = eval(ae.details["GPU"])

        return ae