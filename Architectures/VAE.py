import os
from tensorflow.keras import layers, Model, activations, regularizers, backend as K, ops, losses
import tensorflow as tf


os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers, Model, activations

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding an image."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class VAE:
    def __init__(
            self,
            input_shape,
            halvings=2,
            init_hidden_depth=8,
            latent_dim=10_000,
            output_activation="tanh",
            hidden_activation=activations.leaky_relu
            ):
        
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
        
        autoencoder_input = encoder_input
        z_mean_output, z_log_var_output, z = self.encoder(encoder_input)
        autoencoder_output = self.decoder(z)
        self.autoencoder = Model(autoencoder_input, autoencoder_output, name="autoencoder")

        # Custom loss function with KL divergence
        def vae_loss(x, x_decoded):
            reconstruction_loss = ops.mean(ops.sum(keras.losses.binary_crossentropy(x, x_decoded), axis=(1, 2)))
            kl_loss = -0.5 * ops.sum(1 + z_log_var_output - ops.square(z_mean_output) - ops.exp(z_log_var_output), axis=-1)
            return ops.mean(reconstruction_loss + kl_loss)
        
        self.autoencoder.compile(optimizer=keras.optimizers.Adam(), loss=vae_loss)
        self.autoencoder.summary()
    
    def train(self, x_train, epochs=2, batch_size=16):
        history = self.autoencoder.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.2
        )
        return history
    
    def encode(self, x):
        """Encode input data x into its latent space representation."""
        return self.encoder.predict(x)
    
    def decode(self, y):
        """Decode latent space representation y into the original data space."""
        return self.decoder.predict(y)
