from pydantic import BaseModel
from typing import Optional
import tensorflow as tf
from keras import Input
from keras.models import Model, load_model, save_model
from keras.layers import Reshape, Dense, Input, Activation, BatchNormalization, \
    Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D, \
    Dropout, Add, Flatten
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K


def conv(Inp, num_filters, kernel_size, padding, max_pool, activation):
    conv1 = Conv2D(num_filters, kernel_size, padding=padding)(Inp)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)
    if max_pool:
        conv1 = MaxPooling2D()(conv1)
    return conv1


def convup(Inp, num_filters, kernel_size, padding,
           activation):
    x = Conv2DTranspose(num_filters, kernel_size, padding=padding)(Inp)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = UpSampling2D()(x)
    return x


def sampling(inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def kl_loss_calc(z_log_var, z_mean):
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    return kl_loss


def rec_loss(y_true, y_rec):
    return K.sqrt(K.mean(K.square(y_true - y_rec)))


class EncInputs(BaseModel):
    num_filters: int
    kernel_size: int = 3
    padding: str = 'same'
    activation: str = 'relu'
    max_pool: bool = True


class DecInputs(BaseModel):
    num_filters: int
    kernel_size: int = 3
    padding: str = 'same'
    activation: str = 'relu'


class Train(tf.keras.Model):

    def __init__(self, encoder, decoder, train, val, epochs, lr=1e-3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.val_kl = tf.keras.metrics.Mean(name='val_kl_loss')
        self.val_re = tf.keras.metrics.Mean(name='val_re_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.train_runner(train, val, epochs, lr)

    def call(self, data):
        a, b, c = self.encoder(data)
        d = self.decoder(c)
        return a, b, c, d

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, reconstruction = self(x)
            reconstruction_loss = rec_loss(x, reconstruction)
            kl_loss = kl_loss_calc(z_log_var, z_mean)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

    @tf.function
    def test_step(self, data):
        x, y = data
        z_mean, z_log_var, z, reconstruction = self(x, training=False)
        reconstruction_loss = rec_loss(x, reconstruction)
        kl_loss = kl_loss_calc(z_log_var, z_mean)
        total_loss = reconstruction_loss + kl_loss
        self.val_kl.update_state(kl_loss)
        self.val_re.update_state(reconstruction_loss)
        self.val_loss.update_state(total_loss)

    def train_runner(self, train, val, epochs, lr):
        for t in train:
            break
        batch_size = t[0].shape[0]
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        for epochs in range(epochs):
            batches = 0
            for data in train:
                self.train_step(data)
                batches += 1
                if batches >= train.samples // batch_size:
                    batches = 0
                    break
            for data in val:
                self.test_step(data)
                batches += 1
                if batches > val.samples // batch_size:
                    break
            template = (
                "epoch: {}, train_loss: {},val_loss: {}, val_rec: {}, val_kl: {}")
            print(template.format(epochs, self.total_loss_tracker.result(),
                                  self.val_loss.result(), self.val_re.result(),
                                  self.val_kl.result()))


class VAE:
    def __init__(self, input_shape, encoder_arch: list, latent_dims: int,
                 decoder_arch='encoder'):
        self.encoder = None
        self.decoder = None
        self.train_runner = None
        self.encoder_last_conv_shape = None
        self.input_shape = input_shape
        self.encoder_arch = encoder_arch
        self.latent_dims = latent_dims
        self.decoder_arch = decoder_arch
        self.construct_encoder()
        self.construct_decoder()

    def construct_encoder(self):
        Inp = Input(self.input_shape)
        for i, params in enumerate(self.encoder_arch):
            params = EncInputs(**params).dict()
            if i == 0:
                layer = conv(Inp=Inp, **params)
            else:
                layer = conv(Inp=layer, **params)
        self.encoder_last_conv_shape = layer.shape
        layer = Flatten(name='Flatten_encoder_layer')(layer)
        layer = Dense(self.latent_dims, activation='relu')(layer)
        z_mean = Dense(self.latent_dims, name="z_mean")(layer)
        z_log_var = Dense(self.latent_dims, name="z_log_var")(layer)
        z = sampling([z_mean, z_log_var])
        self.encoder = Model(Inp, [z_mean, z_log_var, z], name="encoder")

        return self.encoder

    def construct_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dims,))
        shp_ = self.encoder.get_layer('Flatten_encoder_layer').output_shape
        fl = Dense(shp_[1])(latent_inputs)
        re = Reshape(self.encoder_last_conv_shape[1:])(fl)
        if self.decoder_arch == 'encoder':
            self.encoder_arch.reverse()
            self.encoder_arch = self.encoder_arch
        #
        for i, params in enumerate(self.encoder_arch):
            params.pop('max_pool', None)
            params = DecInputs(**params).dict()
            if i == 0:
                layer = convup(Inp=re, **params)
            else:
                layer = convup(Inp=layer, **params)
        out = conv(layer, self.input_shape[2], 1, 'same', False, 'relu')
        self.decoder = Model(latent_inputs, out, name='decoder')

# l = [
#     {'num_filters': 24, 'kernel_size': 3},
#     {'num_filters': 120, 'kernel_size': 5}
# ]
#
# obj = VAE((128, 128, 1), l, 20)
# # print(obj.encoder.summary())
# # print(obj.decoder.summary())
