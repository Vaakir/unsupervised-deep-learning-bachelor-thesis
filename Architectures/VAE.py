import os
import pickle
from tensorflow.keras import layers, Model, activations, backend as K, ops, losses
import tensorflow as tf
import numpy as np
import keras
from keras import ops
from keras import layers, Model, activations

os.environ["KERAS_BACKEND"] = "tensorflow"

class Sampling(layers.Layer):
    def call(self, inputs):
        mu, logvar = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * logvar) * epsilon

class VAE(keras.Model):
    def __init__(
            self,
            input_shape,
            halvings=4,
            init_hidden_depth=8,
            latent_dim=1_000,
            output_activation="tanh",
            lambda_=1e-5,
            hidden_activation=activations.leaky_relu
            ):
        super(VAE, self).__init__()
        self.lambda_ = lambda_

        shape_changed=False
        if input_shape[-1]>3:
            input_shape = list(input_shape) + [1]
            shape_changed=True
        if len(input_shape)>4:
            input_shape = input_shape[-4:]
            shape_changed=True
        if shape_changed: print(f"Interpreted image shape: {tuple(input_shape)}")

        # Encoder
        encoder_input = x = layers.Input(shape=input_shape)
        depth = init_hidden_depth
        for _ in range(halvings):
            x = layers.Conv3D(depth, (3, 3, 3), strides=2, activation=hidden_activation, padding="same")(x)
            x = layers.Dropout(0.05)(x)
            depth <<= 1

        pre_flatten_shape = x.shape
        print(f"Pre-flattened latent shape: {pre_flatten_shape}")
        x = layers.Flatten()(x)
        
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        
        self.encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        decoder_input = layers.Input(shape=(latent_dim,))
        x = layers.Dense(pre_flatten_shape[1] * pre_flatten_shape[2] * pre_flatten_shape[3] * pre_flatten_shape[4], activation="relu")(decoder_input)
        x = layers.Reshape(pre_flatten_shape[1:])(x)
        
        for _ in range(halvings):
            depth >>= 1
            x = layers.Conv3D(depth, (3, 3, 3), activation=hidden_activation, padding="same")(x)
            x = layers.UpSampling3D((2, 2, 2))(x)
        decoded = layers.Conv3D(1, (3, 3, 3), activation=output_activation, padding="same")(x)
        self.decoder = Model(decoder_input, decoded, name="decoder")
        
    def call(self, inputs):
        z, mu, logvar = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))*self.lambda_
        self.add_loss(kl_loss)
        return reconstructed
    
    def save(self, path):
        """Save the VAE model to disk."""
        os.makedirs(path, exist_ok=True)
        model_data = {
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
            'lambda_': self.lambda_
        }
        with open(os.path.join(path, "vae_model.pkl"), "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")
    
    @staticmethod
    def open(path):
        """Load the VAE model from disk."""
        model_file = os.path.join(path, "vae_model.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"No saved model found in {path}")
        with open(model_file, "rb") as f:
            model_data = pickle.load(f)
        
        vae_instance = VAE(input_shape=(0, 0, 0))  # Dummy shape to initialize
        vae_instance.encoder = Model.from_config(model_data['encoder'])
        vae_instance.decoder = Model.from_config(model_data['decoder'])
        vae_instance.lambda_ = model_data['lambda_']
        
        print(f"Model loaded from {path}")
        return vae_instance