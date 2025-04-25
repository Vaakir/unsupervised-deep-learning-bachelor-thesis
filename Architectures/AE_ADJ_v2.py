import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, Model
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from Metrics.metrics_tf import MSE_loss, NMSE_loss, NRMSE_loss, SSIM_loss


# Define the sampling function for the reparameterization trick
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, logvar = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * logvar) * epsilon

# Define the VAE model
class AE(keras.Model):
    def __init__(
            self,
            input_shape,
            latent_dim,
            encoder_layers,
            decoder_layers,
            name="AE",
            VAE_model=True,
            beta=1,
            debug=False,
            hidden_activation=activations.leaky_relu,
            loss_fn="mse",
            optimizer="adam",
            ):
        super(AE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.VAE_model = VAE_model
        self.name = name
        self.beta = beta
        
        self.history = None
        self.debug = debug
        self.hidden_activation = hidden_activation
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.training_time = 0
        #self.GPU = GPU

        self.hist = {'loss': [], 'val_loss': [], 'reconstruction_loss': [], 'kl_loss': []}
        
        # shape_changed=False
        #if input_shape[-1]>3:
        #    input_shape = list(input_shape) + [1]
        #    shape_changed=True
        #if len(input_shape)>4:
        #    input_shape = input_shape[-4:]
        #    shape_changed=True
        #if shape_changed: 
        #    print(f"Interpreted image shape: {tuple(input_shape)}")

        if len(self.input_shape) == 4: 
            self.input_shape = (list(self.input_shape) + [1])[-4:] # (8, 80, 96, 80) -> [8, 80, 96, 80, 1] -> [80, 96, 80, 1] 
        if len(self.input_shape) == 3:
            self.input_shape = (list(self.input_shape) + [1])[-3:] # (8, 80, 96) -> [8, 80, 96, 1] -> [80, 96, 1] 
        print(f"Interpreted image shape: {tuple(self.input_shape)}", end=" ")

        # Encoder
        encoder_input_layer = x = layers.Input(shape=self.input_shape)
        x = self.add_layers(x, self.encoder_layers)
        pre_flatten_shape = x.shape
        x = layers.Flatten()(x)
        print(f"Pre-flattened latent shape: {pre_flatten_shape}")
        
        # VAE specific: Output mean and log variance
        if self.VAE_model:
            self.mean = layers.Dense(self.latent_dim, name="mean")(x)
            self.log_var = layers.Dense(self.latent_dim, name="log_var")(x)
            z = Sampling()([self.mean, self.log_var])
            self.encoder = Model(encoder_input_layer, [self.mean, self.log_var, z], name="encoder")
        else:
            x = layers.Dense(self.latent_dim, activation=hidden_activation)(x)
            self.encoder = Model(encoder_input_layer, x, name="encoder")

        # Reshape the latent vector back to the original spatial dimensions
        decoder_input = layers.Input(shape=(latent_dim,))
        x = layers.Dense(np.prod(pre_flatten_shape[1:5]), activation=hidden_activation)(decoder_input)
        x = layers.Reshape(pre_flatten_shape[1:])(x)
        x = self.add_layers(x, self.decoder_layers)
        
        self.decoder = Model(decoder_input, x, name="decoder")

        # 3. Combine the encoder and decoder into a full AE / VAE
        #if self.VAE_model:
        #    mean, log_var, z = self.encoder(encoder_input_layer)
        #    autoencoder_output = self.decoder(z)
        #    self.autoencoder = Model(encoder_input_layer, autoencoder_output, name="autoencoder")
        #    # kl_loss = -0.5 * tf.reduce_mean(
        #    #     1 + log_var - tf.square(mean) - tf.exp(log_var)
        #    # )            
        #    # total_loss = reconstruction_loss + kl_loss
        #else:
        #    autoencoder_output = self.decoder(self.encoder(encoder_input_layer))
        #    self.autoencoder = Model(encoder_input_layer, autoencoder_output, name="autoencoder")
        #
        ##self.autoencoder.compile(optimizer=keras.optimizers.Adam(), loss="mse")
    
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
    
    def train(self, x_train, epochs=2, batch_size=16, verbose=False, save_path="", save_interval=100, patience=100):
        """A seperate train function to save models every n interval and do other usefull modifications"""
        start_time = time.time()        
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        print(f"Training {self.name} on CPU") #{'GPU' if self.GPU else 'CPU'}")
        print(f"Input shape: {x_train.shape}")

        best_val_loss = float('inf')
        wait = 0
        best_model_weights = None  # Store the best model's weights

        for epoch in range(1, epochs + 1):
            history = self.fit(
                x_train, x_train,
                epochs=1,
                batch_size=batch_size,
                shuffle=True,
                validation_split=0.2,
                verbose=verbose
            )
            
            self.hist["loss"].extend(history.history["loss"])
            self.hist["val_loss"].extend(history.history["val_loss"])
            self.history = self.hist

            # usefull if you want to track progress
            if save_path and epoch % save_interval == 0:
                self.save(save_path, name=f"{self.name}", verbose=verbose)
            
            # save weights of the best model
            val_loss = history.history["val_loss"][0]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = self.get_weights()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch} due to no improvement in val_loss for {patience} epochs.")
                    break
        
        # After training, save the best model
        if best_model_weights is not None and save_path:
            self.set_weights(best_model_weights)  # Restore the best model weights
            self.save(save_path, name=f"{self.name}", verbose=verbose)

        self.training_time = time.time() - start_time
        print(f"Training complete in {round(self.training_time, 2)}s")
        return self.history

    def call(self, inputs):
        if self.VAE_model:
            z, mu, logvar = self.encoder(inputs)
            reconstructed = self.decoder(z)
            kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar)) * self.beta
            self.add_loss(kl_loss)
        else:
            reconstructed = self.decoder(self.encoder(inputs))
        return reconstructed
    
    def encode(self, x):
        """Encode input data x into its latent space representation."""
        return self.encoder.predict(x)

    def decode(self, y):
        """Decode latent space representation y into the original data space."""
        return self.decoder.predict(y)
    
    def autoencode(self, x):
        """Encode and decode again"""
        latent = self.encode(x)
        if self.VAE_model:
            latent = latent[2]
        recon = self.decode(latent)
        recon = np.squeeze(recon, axis=-1) #((0,0,0,0)) -> ((0,0,0))
        return recon

    # Sacred meme, do not delete
    # def compile(self):
    #    self.compile(optimizer=self.optimizer, loss=self.loss, metrics=None)
    
    def compile_model(self):
        self.compile(optimizer=self.optimizer, loss=self.loss_fn)
        return self

    def save(self, model_path, name="AE_test", verbose=False):
        """Save the VAE model, encoder, and decoder to disk."""
        model_folder = os.path.join(model_path, f"{name}")
        epoch_folder = os.path.join(model_folder, "epoch_"+str(len(self.hist["loss"])))
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(epoch_folder, exist_ok=True)

        self.encoder.save(os.path.join(epoch_folder, "encoder.keras"))
        self.decoder.save(os.path.join(epoch_folder, "decoder.keras"))
        if verbose:
            print(f"Models saved to {epoch_folder} as '{name}_autoencoder.keras', '{name}_encoder.keras', '{name}_decoder.keras'.")

        
        # Save loss history if available
        if self.history is not None:
            history_path = os.path.join(epoch_folder, "history.json")
            with open(history_path, 'w') as f:
                json.dump(self.hist, f)
            if verbose:
                print(f"Loss history saved to {history_path}")

        configs = {
            "input_shape": str(self.input_shape),
            "latent_dim": str(self.latent_dim),
            "encoder_layers": str(self.encoder_layers),
            "decoder_layers": str(self.decoder_layers),
            "name": str(name),
            "VAE_model": str(self.VAE_model),
            "beta": str(self.beta),
            "training_time": str(self.training_time),

            "loss_fn": str(self.loss_fn) if type(self.loss_fn) is str else str(self.loss_fn.__name__),
            "debug": str(self.debug),
            "hidden_activation": str(self.hidden_activation),
            "optimizer": str(self.optimizer)
        }

        # Save the model details (converted to strings) in a JSON file
        configs_path = os.path.join(epoch_folder, "configs.json")
        with open(configs_path, 'w') as f:
            json.dump(configs, f, indent=4)
        if verbose:
            print(f"Model details saved to {configs_path}")
  
    @staticmethod
    def open(path):
        """Load the VAE model, encoder, and decoder from disk."""
        encoder = tf.keras.models.load_model(os.path.join(path, "encoder.keras"), custom_objects={'Sampling': Sampling})
        decoder = tf.keras.models.load_model(os.path.join(path, "decoder.keras"))
        
        # Create an instance of VAE without initializing models
        ae = AE.__new__(AE)
        ae.encoder = encoder
        ae.decoder = decoder

        with open(os.path.join(path, "history.json")) as f:
            ae.hist = json.load(f) #ae.history = {'loss': [], 'val_loss': []}  
            ae.history = ae.hist
        

        ae.name="ok"
        ae.built = True # I should have used .h5 file types so this small hack wouldnt be needed (it allows me to load and continue training the model for testing)

        print("Models loaded successfully.")

        with open(os.path.join(path, "configs.json")) as f:
            ae.details = json.load(f)
            
            # Ensure these are boolean values, convert them if necessary # [False, True]["False" == "True"]
            ae.input_shape = eval(ae.details["input_shape"])
            ae.latent_dim = eval(ae.details["latent_dim"])
            ae.encoder_layers = ae.details["encoder_layers"]
            ae.decoder_layers = ae.details["decoder_layers"]
            ae.name = ae.details["name"]
            ae.VAE_model = eval(ae.details["VAE_model"])
            ae.beta = eval(ae.details["beta"])
            ae.training_time = eval(ae.details["training_time"]) if "training_time" in ae.details else 0
            ae.loss_fn = {"MSE_loss": MSE_loss, "SSIM_loss": SSIM_loss, "NMSE_loss": NMSE_loss, "NRMSE_loss": NRMSE_loss}[ae.details["loss_fn"]] 
            ae.debug = eval(ae.details["debug"])
            
            ae.hidden_activation = str(ae.details["hidden_activation"]), #eval(ae.details["hidden_activation"]),
            ae.optimizer = str(ae.details["optimizer"]),
        
        # Now reassemble the full VAE model and compile
        #vae.full_model = vae.create_full_model()  # Assuming you have a method to create the full model
        #vae.full_model.compile(optimizer='adam')  # Compile the full model after loading components
        #vae.compile(optimizer="adam")
        #vae._compile_loss = None
        #vae._compile_metrics = None
        #vae._loss_tracker = None
        #super(VAE, vae).compile(optimizer="adam", loss="mse", metrics=None)

        return ae

"""
from tensorflow import keras
from tensorflow.keras import layers, activations, Model
import tensorflow as tf
import numpy as np
import os
import time
import json

# Define the sampling function for the reparameterization trick
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, logvar = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * logvar) * epsilon

# Define the VAE model
class AE(keras.Model):
    def __init__(
            self,
            input_shape,
            latent_dim,
            encoder_layers,
            decoder_layers,
            name="AE",
            VAE_model=True,
            learning_rate=1e-3,
            output_activation="tanh",
            beta=1e-5,
            debug=False,
            GPU=False,
            hidden_activation=activations.leaky_relu
            ):
        super(AE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.VAE_model = VAE_model
        self.name = name
        self.lambda_ = beta
        self.learning_rate = learning_rate
        #self.loss_fn = loss_fn
        
        self.history = None
        self.debug = debug
        self.GPU = GPU
        self.hist = {'loss': [], 'val_loss': [], 'reconstruction_loss': [], 'kl_loss': []}
        
        # shape_changed=False
        #if input_shape[-1]>3:
        #    input_shape = list(input_shape) + [1]
        #    shape_changed=True
        #if len(input_shape)>4:
        #    input_shape = input_shape[-4:]
        #    shape_changed=True
        #if shape_changed: 
        #    print(f"Interpreted image shape: {tuple(input_shape)}")

        if len(self.input_shape) == 4: 
            self.input_shape = (list(self.input_shape) + [1])[-4:] # (8, 80, 96, 80) -> [8, 80, 96, 80, 1] -> [80, 96, 80, 1] 
        if len(self.input_shape) == 3:
            self.input_shape = (list(self.input_shape) + [1])[-3:] # (8, 80, 96) -> [8, 80, 96, 1] -> [80, 96, 1] 
        print(f"Interpreted image shape: {tuple(self.input_shape)}", end=" ")

        # Encoder
        encoder_input_layer = x = layers.Input(shape=self.input_shape)
        x = self.add_layers(x, self.encoder_layers)
        pre_flatten_shape = x.shape
        x = layers.Flatten()(x)
        print(f"Pre-flattened latent shape: {pre_flatten_shape}")
        
        # VAE specific: Output mean and log variance
        if self.VAE_model:
            self.mean = layers.Dense(self.latent_dim, name="mean")(x)
            self.log_var = layers.Dense(self.latent_dim, name="log_var")(x)
            z = Sampling()([self.mean, self.log_var])
            self.encoder = Model(encoder_input_layer, [self.mean, self.log_var, z], name="encoder")
        else:
            x = layers.Dense(self.latent_dim, activation="leaky_relu")(x)
            self.encoder = Model(encoder_input_layer, x, name="encoder")

        # Reshape the latent vector back to the original spatial dimensions
        decoder_input = layers.Input(shape=(latent_dim,))
        x = layers.Dense(np.prod(pre_flatten_shape[1:5]), activation="relu")(decoder_input)
        x = layers.Reshape(pre_flatten_shape[1:])(x)
        x = self.add_layers(x, self.decoder_layers)
        
        self.decoder = Model(decoder_input, x, name="decoder")

        # 3. Combine the encoder and decoder into a full AE / VAE
        #if self.VAE_model:
        #    mean, log_var, z = self.encoder(encoder_input_layer)
        #    autoencoder_output = self.decoder(z)
        #    self.autoencoder = Model(encoder_input_layer, autoencoder_output, name="autoencoder")
        #    # kl_loss = -0.5 * tf.reduce_mean(
        #    #     1 + log_var - tf.square(mean) - tf.exp(log_var)
        #    # )            
        #    # total_loss = reconstruction_loss + kl_loss
        #else:
        #    autoencoder_output = self.decoder(self.encoder(encoder_input_layer))
        #    self.autoencoder = Model(encoder_input_layer, autoencoder_output, name="autoencoder")
        #
        ##self.autoencoder.compile(optimizer=keras.optimizers.Adam(), loss="mse")
    
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
    
    def train(self, x_train, epochs=2, batch_size=16, verbose=False, save_path="", save_interval=100):
        "A seperate train function to save models every n interval and do other usefull modifications"
        start_time = time.time()        
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        print(f"Training {self.name} on {'GPU' if self.GPU else 'CPU'}")
        print(f"Input shape: {x_train.shape}")

        for epoch in range(1, epochs + 1):
            history = self.fit(
                x_train, x_train,
                epochs=1,
                batch_size=batch_size,
                shuffle=True,
                validation_split=0.2,
                verbose=verbose
            )
            
            self.hist["loss"].extend(history.history["loss"])
            self.hist["val_loss"].extend(history.history["val_loss"])
            self.history = self.hist

            if save_path and epoch % save_interval == 0:
                #self.save(os.path.join(save_path, name=f"{self.name}_epoch{epoch}"))
                self.save(save_path, name=f"{self.name}")

        self.training_time = time.time() - start_time
        print(f"Training complete in {round(self.training_time, 2)}s")
        return self.history

    def call(self, inputs):
        if self.VAE_model:
            z, mu, logvar = self.encoder(inputs)
            reconstructed = self.decoder(z)
            kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))*self.lambda_
            self.add_loss(kl_loss)
        else:
            reconstructed = self.decoder(self.encoder(inputs))
        return reconstructed
    
    def encode(self, x):
        "Encode input data x into its latent space representation."
        return self.encoder.predict(x)

    def decode(self, y):
        "Decode latent space representation y into the original data space."
        return self.decoder.predict(y)
    
    def autoencode(self, x):
        latent = self.encode(x)
        if self.VAE_model:
            latent = latent[2]
        recon = self.decode(latent)
        recon = np.squeeze(recon, axis=-1) #((0,0,0,0)) -> ((0,0,0))
        return recon

    def save(self, model_path, name):
        "Save the VAE model, encoder, and decoder to disk."
        model_folder = os.path.join(model_path, f"{name}")
        epoch_folder = os.path.join(model_folder, "epoch_"+str(len(self.hist["loss"])))
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(epoch_folder, exist_ok=True)

        self.encoder.save(os.path.join(epoch_folder, "encoder.keras"))
        self.decoder.save(os.path.join(epoch_folder, "decoder.keras"))
        print(f"Models saved to {epoch_folder} as '{name}_autoencoder.keras', '{name}_encoder.keras', '{name}_decoder.keras'.")

        
        # Save loss history if available
        if self.history is not None:
            history_path = os.path.join(epoch_folder, "history.json")
            with open(history_path, 'w') as f:
                json.dump(self.hist, f)
            print(f"Loss history saved to {history_path}")

        configs = {
            "input_shape": str(self.input_shape),
            "latent_dim": str(self.latent_dim),
            "encoder_layers": str(self.encoder_layers),
            "decoder_layers": str(self.decoder_layers),
            "name": str(name),
            #"loss": str(self.loss_fn.__name__),
            "learning_rate": str(self.learning_rate),
            "VAE_model": str(self.VAE_model),
            "debug": str(self.debug),
            "GPU": str(self.GPU)
        }

        # Save the model details (converted to strings) in a JSON file
        configs_path = os.path.join(epoch_folder, "configs.json")
        with open(configs_path, 'w') as f:
            json.dump(configs, f, indent=4)
        print(f"Model details saved to {configs_path}")
  
    @staticmethod
    def open(path):
        "Load the VAE model, encoder, and decoder from disk."
        encoder = tf.keras.models.load_model(os.path.join(path, "encoder.keras"), custom_objects={'Sampling': Sampling})
        decoder = tf.keras.models.load_model(os.path.join(path, "decoder.keras"))
        
        # Create an instance of VAE without initializing models
        ae = AE.__new__(AE)
        ae.encoder = encoder
        ae.decoder = decoder

        with open(os.path.join(path, "history.json")) as f:
            ae.hist = json.load(f) #ae.history = {'loss': [], 'val_loss': []}  
            ae.history = ae.hist

        ae.name="ok"
        ae.built = True

        print("Models loaded successfully.")

        with open(os.path.join(path, "configs.json")) as f:
            ae.details = json.load(f)
            
            # Ensure these are boolean values, convert them if necessary # [False, True]["False" == "True"]
            ae.input_shape = eval(ae.details["input_shape"])
            ae.latent_dim = eval(ae.details["latent_dim"])
            ae.encoder_layers = ae.details["encoder_layers"]
            ae.decoder_layers = ae.details["decoder_layers"]
            ae.name = ae.details["name"]
            #vae.loss = {"MSE_loss": MSE_loss, "SSIM_loss": SSIM_loss}[vae.details["loss"]]
            # ae.loss = ae.details["loss"]
            ae.learning_rate = eval(ae.details["learning_rate"])
            ae.VAE_model = eval(ae.details["VAE_model"])
            ae.debug = eval(ae.details["debug"])
            ae.GPU = eval(ae.details["GPU"])

        # Now reassemble the full VAE model and compile
        #vae.full_model = vae.create_full_model()  # Assuming you have a method to create the full model
        #vae.full_model.compile(optimizer='adam')  # Compile the full model after loading components
        #vae.compile(optimizer="adam")
        #vae._compile_loss = None
        #vae._compile_metrics = None
        #vae._loss_tracker = None
        #super(VAE, vae).compile(optimizer="adam", loss="mse", metrics=None)

        return ae
"""