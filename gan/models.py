# gan/models.py

"""
Lightweight model builders for the Conditional DCGAN.

Provides:
    - build_generator(latent_dim, num_classes, img_shape)
    - build_discriminator(img_shape, num_classes)
    - build_models(latent_dim, num_classes, img_shape, lr=2e-4, beta_1=0.5)
      -> {"generator": G, "discriminator": D, "gan": combined}

Notes
-----
- Compatible with Keras 3 (no legacy optimizers).
- LeakyReLU uses `negative_slope=0.2` to avoid deprecation warnings.
- Generator outputs images in [-1, 1] (tanh), matching the training loop.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Use standard (non-legacy) Adam for Keras 3
Adam = tf.keras.optimizers.Adam


# -------------------------------------------------
# Generator
# -------------------------------------------------
def build_generator(latent_dim: int = 100,
                    num_classes: int = 9,
                    img_shape: Tuple[int, int, int] = (40, 40, 1)) -> tf.keras.Model:
    """
    Conditional generator: concatenates a spatially-projected label map to the noise feature map.
    Output: tanh -> [-1, 1]
    """
    H, W, C = img_shape

    z_in = layers.Input(shape=(latent_dim,), name="z")
    y_in = layers.Input(shape=(num_classes,), name="y_onehot")

    # Project noise
    x = layers.Dense(5 * 5 * 256, use_bias=False)(z_in)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape((5, 5, 256))(x)

    # Project label into a single channel map at the same spatial size
    y_map = layers.Dense(5 * 5 * 1, use_bias=False)(y_in)
    y_map = layers.Reshape((5, 5, 1))(y_map)

    # Concatenate noise feature map with label map
    x = layers.Concatenate(axis=-1)([x, y_map])  # (5,5,257)

    # Upsampling blocks to reach (40,40)
    x = layers.UpSampling2D()(x)  # 10x10
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="gen_conv1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.UpSampling2D()(x)  # 20x20
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="gen_conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.UpSampling2D()(x)  # 40x40
    out = layers.Conv2D(C, 3, padding="same", activation="tanh", use_bias=False, name="gen_out")(x)

    return models.Model([z_in, y_in], out, name="Conditional_Generator")


# -------------------------------------------------
# Discriminator
# -------------------------------------------------
def build_discriminator(img_shape: Tuple[int, int, int] = (40, 40, 1),
                        num_classes: int = 9) -> tf.keras.Model:
    """
    Conditional discriminator: concatenates the input image with a label-projected map.
    Output: single sigmoid logit (real/fake).
    """
    H, W, C = img_shape

    x_in = layers.Input(shape=img_shape, name="x")
    y_in = layers.Input(shape=(num_classes,), name="y_onehot")

    # Project label into an image-sized map and concatenate as an extra channel
    y_map = layers.Dense(H * W * C)(y_in)
    y_map = layers.Reshape(img_shape)(y_map)

    xy = layers.Concatenate(axis=-1)([x_in, y_map])  # (H,W,C+1)

    x = layers.Conv2D(64, 4, strides=2, padding="same", name="disc_conv1")(xy)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same", name="disc_conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(1, activation="sigmoid", name="disc_out")(x)

    return models.Model([x_in, y_in], out, name="Conditional_Discriminator")


# -------------------------------------------------
# Builder for D, G, and combined GAN
# -------------------------------------------------
def build_models(latent_dim: int,
                 num_classes: int,
                 img_shape: Tuple[int, int, int],
                 lr: float = 2e-4,
                 beta_1: float = 0.5) -> Dict[str, tf.keras.Model]:
    """
    Create and compile:
        - Discriminator D (trainable alone, binary_crossentropy)
        - Generator G (uncompiled; trained via combined)
        - Combined GAN = D(G(z,y), y) with D frozen (binary_crossentropy)
    """
    # Models
    G = build_generator(latent_dim=latent_dim, num_classes=num_classes, img_shape=img_shape)
    D = build_discriminator(img_shape=img_shape, num_classes=num_classes)

    # Optimizers (Keras 3 compatible)
    d_opt = Adam(learning_rate=lr, beta_1=beta_1)
    g_opt = Adam(learning_rate=lr, beta_1=beta_1)

    # Compile discriminator
    D.compile(optimizer=d_opt, loss="binary_crossentropy", metrics=["accuracy"])

    # Build combined GAN with D frozen
    z_in = layers.Input(shape=(latent_dim,), name="z_in")
    y_in = layers.Input(shape=(num_classes,), name="y_in")
    fake = G([z_in, y_in])
    D.trainable = False
    validity = D([fake, y_in])
    GAN = models.Model([z_in, y_in], validity, name="Conditional_GAN")
    GAN.compile(optimizer=g_opt, loss="binary_crossentropy")

    return {"generator": G, "discriminator": D, "gan": GAN}

