"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, Lambda, Input,
    MaxPooling2D, concatenate, BatchNormalization,
    GlobalAveragePooling2D, GlobalMaxPool2D, Dense, Flatten
)


def get_model_definition(img_height, img_width, in_channels, out_channels, dim):
    """
    Defines U-Net like, TensorFlow model

    Parameters
    ----------
    img_height : Height of the input image
    img_width : Width of the input image
    in_channels : Number of channels of the input image
    out_channels : Output channels of the model, i.e. the number of classes of the result.

    Returns
    -------
    TensorFlow model
    """
    base = (2 ** 3)
    assert img_height % base == 0, f"{img_height} not multiple of {base}"
    assert img_width % base == 0, f"{img_width} not multiple of {base}"
    inputs = Input((img_height, img_width, in_channels))

    kwargs_conv2d = {
        'kernel_size': (3, 3),
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'padding': 'same',
    }
    outs = {
        1: dim,
        2: 2 * dim,
        3: 4 * dim,
        4: 8 * dim,
        # 5: dim,
    }
    pre_processed = Lambda(lambda x: x / 255)(inputs)
    """
    Tile model
    """
    # Down
    c1 = Conv2D(outs[1], **kwargs_conv2d)(pre_processed)
    c1 = Conv2D(outs[1], **kwargs_conv2d)(c1)
    c2 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(outs[2], **kwargs_conv2d)(c2)
    c2 = Conv2D(outs[2], **kwargs_conv2d)(c2)
    c3 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(outs[3], **kwargs_conv2d)(c3)
    c3 = Conv2D(outs[3], **kwargs_conv2d)(c3)
    c4 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(outs[4], **kwargs_conv2d)(c4)
    features = Conv2D(outs[4], **kwargs_conv2d)(c4)
    outputs_tile = GlobalMaxPool2D()(features)
    outputs_tile = Dense(dim, activation='relu')(outputs_tile)
    outputs_tile = Dense(dim, activation='relu')(outputs_tile)
    outputs_tile = Dense(out_channels, activation='sigmoid')(outputs_tile)
    """
    Pixel model
    """
    # Down
    c1 = Conv2D(outs[1], **kwargs_conv2d)(pre_processed)
    c1 = Conv2D(outs[1], **kwargs_conv2d)(c1)
    c2 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(outs[2], **kwargs_conv2d)(c2)
    c2 = Conv2D(outs[2], **kwargs_conv2d)(c2)
    c3 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(outs[3], **kwargs_conv2d)(c3)
    c3 = Conv2D(outs[3], **kwargs_conv2d)(c3)
    c4 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(outs[4], **kwargs_conv2d)(c4)
    c4 = Conv2D(outs[4], **kwargs_conv2d)(c4)
    # Up
    c4 = concatenate([features, c4])
    u3 = Conv2DTranspose(outs[3], (2, 2), strides=(2, 2), padding='same')(c4)
    u3 = concatenate([u3, c3])
    u3 = Conv2D(outs[3], **kwargs_conv2d)(u3)
    u3 = Conv2D(outs[3], **kwargs_conv2d)(u3)
    u2 = Conv2DTranspose(outs[2], (2, 2), strides=(2, 2), padding='same')(u3)
    u2 = concatenate([u2, c2])
    u2 = Conv2D(outs[2], **kwargs_conv2d)(u2)
    u2 = Conv2D(outs[2], **kwargs_conv2d)(u2)
    u1 = Conv2DTranspose(outs[1], (2, 2), strides=(2, 2), padding='same')(u2)
    u1 = concatenate([u1, c1], axis=3)
    u1 = Conv2D(outs[1], **kwargs_conv2d)(u1)
    outputs_pixel = Conv2D(out_channels, (1, 1), activation='sigmoid')(u1)
    # Model compilation
    tile_model = Model(inputs=[inputs], outputs=[outputs_tile])
    pixel_model = Model(inputs=[inputs], outputs=[outputs_pixel])
    return tile_model, pixel_model
