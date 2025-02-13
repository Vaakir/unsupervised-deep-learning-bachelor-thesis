from tensorflow.keras import layers, Model, activations, regularizers
import tensorflow as tf
import os

class AE:
    def __init__(
            self,
            input_shape,
            halvings=2,
            init_hidden_depth:int=8,
            hidden_depth_grow_factor:int=2,

            latent_dim=10_000,
            output_activation="tanh",
            hidden_activation=activations.leaky_relu,
            l1_lambda=0.0000001  # L1 regularization factor
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
            depth *= hidden_depth_grow_factor

        # Flatten the latent space
        pre_flatten_shape = x.shape
        print(f"Pre-flattened latent shape: {pre_flatten_shape}")
        x = layers.Flatten()(x)
        encoded = layers.Dense(latent_dim, activation="relu", kernel_regularizer=regularizers.l1(l1_lambda))(x)
        self.encoder = Model(encoder_input, encoded, name="encoder")

        # Decoder
        decoder_input = x = layers.Input(shape=encoded.shape[1:])  # Adjusted latent space shape
        
        # Reshape back to 3D
        x = layers.Dense(
            pre_flatten_shape[1] * pre_flatten_shape[2] * pre_flatten_shape[3] * pre_flatten_shape[4],
            activation="relu",
            kernel_regularizer=regularizers.l1(l1_lambda)
        )(decoder_input)
        x = layers.Reshape(pre_flatten_shape[1:])(x)
        
        for _ in range(halvings):
            depth //= hidden_depth_grow_factor
            x = layers.Conv3D(depth, (3, 3, 3), activation=hidden_activation, padding="same")(x)
            x = layers.UpSampling3D((2, 2, 2))(x)  # Single upscale block
        decoded = layers.Conv3D(1, (3, 3, 3), activation=output_activation, padding="same")(x)
        self.decoder = Model(decoder_input, decoded, name="decoder")

        # Full Autoencoder (combine encoder and decoder)
        autoencoder_input = encoder_input
        autoencoder_output = self.decoder(self.encoder(encoder_input))
        self.autoencoder = Model(autoencoder_input, autoencoder_output, name="autoencoder")

        # Compile the autoencoder
        self.autoencoder.compile(loss='mean_squared_error', optimizer='adam')
        self.autoencoder.summary()
        
        self.history = None

    def train(self, x_train, epochs=2, batch_size=16,verbose=True):
        self.history = self.autoencoder.fit(
            x_train, x_train,
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.2
        )
        return self.history

    def encode(self, x):
        """Encode input data x into its latent space representation."""
        return self.encoder.predict(x)

    def decode(self, y):
        """Decode latent space representation y into the original data space."""
        return self.decoder.predict(y)
    
    def save(self, path):
        """Save the autoencoder, encoder, and decoder models to disk."""
        os.makedirs(path, exist_ok=True)
        self.autoencoder.save(os.path.join(path, "autoencoder.keras"))
        self.encoder.save(os.path.join(path, "encoder.keras"))
        self.decoder.save(os.path.join(path, "decoder.keras"))
        print(f"Models saved to {path}")
    
    @staticmethod
    def open(path):
        """Load the autoencoder, encoder, and decoder models from disk."""
        autoencoder = tf.keras.models.load_model(os.path.join(path, "autoencoder.keras"))
        encoder = tf.keras.models.load_model(os.path.join(path, "encoder.keras"))
        decoder = tf.keras.models.load_model(os.path.join(path, "decoder.keras"))

        # Create an instance of AE without initializing models
        ae = AE.__new__(AE)
        ae.autoencoder = autoencoder
        ae.encoder = encoder
        ae.decoder = decoder

        # Compile the loaded model manually
        ae.autoencoder.compile(loss='mean_squared_error', optimizer='adam')

        print("Models loaded and compiled successfully.")
        return ae