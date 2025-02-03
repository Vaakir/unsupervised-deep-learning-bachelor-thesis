from tensorflow.keras import layers, Model, activations, Input
import tensorflow as tf

class GAN:
    def __init__(
            self,
            input_shape,
            output_activation="tanh",
            hidden_activation=activations.leaky_relu
            ):
        
        shape_changed = False
        if input_shape[-1] > 3:
            input_shape = list(input_shape) + [1]
            shape_changed = True
        if len(input_shape) > 4:
            input_shape = input_shape[-4:]
            shape_changed = True
        if shape_changed: 
            print(f"Interpreted image shape: {tuple(input_shape)}")

        self.input_shape = tuple(input_shape)
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan_model = self.define_gan_model(self.generator, self.discriminator)
        
        # Compile the model
        self.gan_model.compile(loss='mean_squared_error', optimizer='adam')
        self.gan_model.summary()

    def build_discriminator(self, trainable=True):
        gamma_init = tf.random_normal_initializer(1., 0.02)
        inp = Input(shape=self.input_shape)
        
        x = layers.Conv3D(64, (4,4,4), strides=2, padding='same', kernel_initializer='he_normal')(inp)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        for i in range(1, 6):
            x = layers.Conv3D(64 * (2 ** i), (4,4,4), strides=2, padding='same', kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv3D(1, (3,3,3), strides=1, padding='same', kernel_initializer='he_normal')(x)
        model = Model(inputs=inp, outputs=x,name="discriminator")
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def build_generator(self, trainable=True):
        gamma_init = tf.random_normal_initializer(1., 0.02)
        inp = Input(shape=self.input_shape)

        depth = 8
        x = layers.Conv3D(depth, (4, 4, 4), strides=2, activation=self.hidden_activation, padding="same", kernel_initializer='he_normal')(inp)
        x = layers.Dropout(0.05)(x)
        
        for _ in range(3):
            x = layers.Conv3D(depth, (4, 4, 4), strides=2, padding="same", kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            depth <<= 1
        
        for _ in range(4):
            depth >>= 1
            x = layers.Conv3DTranspose(depth, (4, 4, 4), strides=2, padding='same', kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization(trainable=trainable)(x)
            x = layers.Activation('relu')(x)
        
        out = layers.Conv3D(1, (1, 1, 1), activation=self.output_activation, padding='same', kernel_initializer='he_normal')(x)
        model = Model(inputs=inp, outputs=out,name="generator")
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def define_gan_model(self, gen_model, dis_model):
        dis_model.trainable = False
        inp = Input(shape=self.input_shape)
        out_g = gen_model(inp)
        out_dis = dis_model(out_g)
        model = Model(inputs=inp, outputs=[out_dis, out_g],name="GAN")
        return model

    def train(self, x_train, epochs=2, batch_size=16):
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                real_images = x_train[i:i+batch_size]
                noise = tf.random.normal([batch_size] + list(self.input_shape))
                fake_images = self.generator(noise)
                
                real_labels = tf.ones((batch_size, 1))
                fake_labels = tf.zeros((batch_size, 1))
                
                d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                
                noise = tf.random.normal([batch_size] + list(self.input_shape))
                g_loss = self.gan_model.train_on_batch(noise, [tf.ones((batch_size, 1)), noise])
                
            print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss}, G Loss: {g_loss}")