from tensorflow.keras import layers, Model, activations

class AE:
    def __init__(self, input_shape, halvings=2, latent_dim=10_000, output_activation="tanh", hidden_activation=activations.leaky_relu):
        
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
        for _ in range(halvings):
            x = layers.Conv3D(10, (3, 3, 3), strides=2, activation=hidden_activation, padding="same")(x)
            x = layers.Dropout(0.05)(x)
        # encoded = layers.Conv3D(8, (3, 3, 3), activation=hidden_activation, padding="same")(x)  # Latent space
        pre_flatten_shape = x.shape
        print(f"Pre-flattened latent shape: {pre_flatten_shape}")
        x = layers.Flatten()(x)
        encoded = layers.Dense(latent_dim, activation="relu")(x)  # Latent space
        self.encoder = Model(encoder_input, encoded, name="encoder")

        self.encoder = Model(encoder_input, encoded, name="encoder")

        # Decoder
        decoder_input = x = layers.Input(shape=encoded.shape[1:])  # Adjusted latent space shape
        
        x = layers.Dense(pre_flatten_shape[1] * pre_flatten_shape[2] * pre_flatten_shape[3]* pre_flatten_shape[4], activation="relu")(decoder_input)

        # Reshape back to 3D volume
        x = layers.Reshape(pre_flatten_shape[1:])(x)
        
        for _ in range(halvings):
            x = layers.Conv3D(10, (3, 3, 3), activation=hidden_activation, padding="same")(x)
            x = layers.UpSampling3D((2, 2, 2))(x)  # Single upscale block

        decoded = layers.Conv3D(1, (3, 3, 3), activation=output_activation, padding="same")(x)  # Final output layer

        self.decoder = Model(decoder_input, decoded, name="decoder")

        # Full Autoencoder (combine encoder and decoder)
        autoencoder_input = encoder_input
        autoencoder_output = self.decoder(self.encoder(encoder_input))
        self.autoencoder = Model(autoencoder_input, autoencoder_output, name="autoencoder")

        # Compile the autoencoder
        self.autoencoder.compile(loss='mean_squared_error', optimizer='adam')
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
