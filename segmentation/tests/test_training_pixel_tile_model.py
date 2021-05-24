"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import cv2
import numpy as np
from loguru import logger
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from cv.tf_utils import train_model
from cv.image_processing import image2tiles, get_labels_tiles
from segmentation.pixel_tile_segmentation_model import get_model_definition


def test_training_pixel_tile_model():
    folder = "./test_data/"
    x = "top_mosaic_09cm_area1_image_resized_crop2.tif"
    y = "top_mosaic_09cm_area1_label_resized_crop2.tif"
    x = f"{folder}/{x}"
    x = cv2.imread(x)
    assert x is not None
    h = 200
    w = 200
    x = image2tiles(x, h, w)
    x = [t[np.newaxis, ...] for t in x]
    x = np.concatenate(x, axis=0)
    y = f"{folder}/{y}"
    y = cv2.imread(y)
    assert y is not None
    y = image2tiles(y, h, w)
    y = get_labels_tiles(y)
    y = y.any(axis=(1, 2)).astype(np.float64)
    assert not np.array_equal(y, np.array((1, 1, 0, 0, 1)).astype(np.float64))
    model_folder = "./tmp/tile_model/"
    model_file = f"{model_folder}/tile_model.save"
    params = {
        'epochs': 3,
        'model_folder': model_folder,
        'model_file': model_file,
    }
    features_model, tile_model, pixel_model = get_model_definition(
        img_height=h, img_width=w, in_channels=3, out_channels=5)
    # tile_model.trainable = False
    tile_model.compile(
        optimizer='adam',
        loss="mse",
        metrics=['mae'])
    model_weights_list_before = tile_model.get_weights()
    logger.info(f"y: {y}")
    y_pred = tile_model.predict(x)
    e1 = np.abs(y - y_pred).mean()
    logger.info(f"e1: {e1}")
    tile_model = train_model(x, y, x, y, tile_model, params, logger)
    y_pred = tile_model.predict(x)
    e2 = np.abs(y - y_pred).mean()
    logger.info(f"e2: {e2}")
    assert e2 < e1
    model_weights_list_after = tile_model.get_weights()
    equals = []
    for weights_before, weights_after in zip(model_weights_list_before, model_weights_list_after):
        equals.append(np.array_equal(weights_before, weights_after))
    assert not all(equals)
