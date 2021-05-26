"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import cv2
import numpy as np
from loguru import logger

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
    y_pixel = get_labels_tiles(y)
    y_tile = y_pixel.any(axis=(1, 2)).astype(np.float64)
    assert not np.array_equal(y, np.array((1, 1, 0, 0, 1)).astype(np.float64))
    params_tile = {
        'epochs': 5,
        'model_folder': "./tmp/tile_model/",
        'model_file': "./tmp/tile_model/tile_model.save",
    }
    tile_model, pixel_model = get_model_definition(
        img_height=h, img_width=w, in_channels=3, out_channels=5, dim=8)
    # tile_model.trainable = False
    tile_model.compile(
        optimizer='adam',
        loss="mse",
        metrics=['mae'])
    model_weights_tile_before = tile_model.get_weights()
    logger.info(f"y_tile: {y_tile}")
    y_pred = tile_model.predict(x)
    e1 = np.abs(y_tile - y_pred).mean()
    tile_model = train_model(x, y_tile, x, y_tile, tile_model, params_tile, logger)
    tile_model.trainable = False
    pixel_model.compile(
        optimizer='adam',
        loss="mse",
        metrics=['mae'])
    y_pred = tile_model.predict(x)
    e2 = np.abs(y_tile - y_pred).mean()
    logger.info(f"e1: {e1}")
    logger.info(f"e2: {e2}")
    assert e2 < e1
    model_weights_tile_after = tile_model.get_weights().copy()
    equals = []
    for weights_before, weights_after in zip(model_weights_tile_before, model_weights_tile_after):
        equals.append(np.array_equal(weights_before, weights_after))
    assert not all(equals)
    model_weights_pixel_before = pixel_model.get_weights()
    y_pred_pixel = pixel_model.predict(x)
    e1 = np.abs(y_pixel - y_pred_pixel).mean()
    params_pixel = {
        'epochs': 10,
        'model_folder': "./tmp/pixel_model/",
        'model_file': "./tmp/pixel_model/pixel_model.save",
    }
    pixel_model = train_model(x, y_pixel, x, y_pixel, pixel_model, params_pixel, logger)
    y_pred_pixel = pixel_model.predict(x)
    e2 = np.abs(y_pixel - y_pred_pixel).mean()
    logger.info(f"e1: {e1}")
    logger.info(f"e2: {e2}")
    assert e2 < e1
    model_weights_pixel_after = pixel_model.get_weights().copy()
    equals = []
    for weights_before, weights_after in zip(model_weights_pixel_before, model_weights_pixel_after):
        equals.append(np.array_equal(weights_before, weights_after))
    assert not all(equals)
    # Comparison of first layer of both models
    tile_weights_first_layer = model_weights_tile_after[0]
    pixel_weights_first_layer = model_weights_pixel_after[0]
    assert np.array_equal(tile_weights_first_layer, pixel_weights_first_layer)
