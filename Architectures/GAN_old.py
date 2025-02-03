import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
# Generator: Encodes an image to a latent space and can also decode from latent space
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(28 * 28 * 1, activation='sigmoid'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

# Discriminator: Classifies images as real or fake
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# GAN model: Combine generator and discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

# Class for the Image Compression GAN
class GAN:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.generator = build_generator(latent_dim)
        self.discriminator = build_discriminator()
        self.gan = build_gan(self.generator, self.discriminator)

        # Compile the discriminator
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Compile the GAN model
        self.gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, x_train, epochs=100, batch_size=128):
        half_batch = batch_size // 2

        for epoch in range(epochs):
            # Train the discriminator
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            real_images = x_train[idx]
            noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
            generated_images = self.generator.predict(noise)

            real_labels = np.ones((half_batch, 1))
            fake_labels = np.zeros((half_batch, 1))

            # Train on real images
            d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
            # Train on fake images
            d_loss_fake = self.discriminator.train_on_batch(generated_images, fake_labels)

            # Train the generator via the GAN model
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            valid_labels = np.ones((batch_size, 1))
            g_loss = self.gan.train_on_batch(noise, valid_labels)

            print(f"{epoch}/{epochs} [D loss: {0.5 * (d_loss_real[0] + d_loss_fake[0])}] [G loss: {g_loss[0]}]")

    def encode(self, images):
        # Use the generator to map images to latent space (compression)
        return self.generator.predict(images)

    def decode(self, latent_vectors):
        # Decode the latent vectors back to images
        return self.generator.predict(latent_vectors)