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


class VaeInputs(BaseModel):
    num_filters: int
    kernel_size: int = 3
    padding: str = 'same'
    activation: str = 'relu'
    max_pool: bool = True


class VaeModel:
    pass


class VAE:
    def __init__(self, input_shape, encoder_arch: list, latent_dims: int,
                 decoder_arch='encoder'):
        self.encoder = None
        self.input_shape = input_shape
        self.encoder_arch = encoder_arch
        self.latent_dims = latent_dims
        self.construct_encoder()

    def construct_encoder(self):
        Inp = Input(self.input_shape)
        for i, params in enumerate(self.encoder_arch):
            params = VaeInputs(**params).dict()
            if i == 0:
                layer = conv(Inp=Inp, **params)
            else:
                layer = conv(Inp=layer, **params)
        layer = Flatten()(layer)
        layer = Dense(self.latent_dims, activation='relu')(layer)
        self.encoder = Model(Inp, layer, name='encoder')
        return self.encoder


l = [
    {'num_filters': 24, 'kernel_size': 3},
    {'num_filters': 120, 'kernel_size': 5}
]

obj = VAE((128, 128, 1), l, 20)
print(obj.encoder.summary())
