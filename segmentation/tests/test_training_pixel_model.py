"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import cv2
import numpy as np
from loguru import logger

from cv.tf_utils import train_model
from cv.image_processing import image2tiles, get_labels_tiles
from segmentation.pixel_segmentation_model import get_model_definition


def test_training_pixel_model():
    folder = "./test_data/"
    x = "top_mosaic_09cm_area1_image_resized.tif"
    x = f"{folder}/{x}"
    x = cv2.imread(x)
    assert x is not None
    h = 200
    w = 200
    x = image2tiles(x, h, w)
    x = [t[np.newaxis, ...] for t in x]
    x = np.concatenate(x, axis=0)
    y = "top_mosaic_09cm_area1_label_resized.tif"
    y = f"{folder}/{y}"
    y = cv2.imread(y)
    assert y is not None
    y = image2tiles(y, h, w)
    y = get_labels_tiles(y)
    model_folder = "./tmp/pixel_model/"
    model_file = f"{model_folder}/pixel_model.save"
    params = {
        'epochs': 3,
        'model_folder': model_folder,
        'model_file': model_file,
    }
    model = get_model_definition(img_height=h, img_width=w, in_channels=3, out_channels=5, dim=8)
    model_weights_list_before = model.get_weights()
    model = train_model(x, y, x, y, model, params, logger)
    model_weights_list_after = model.get_weights()
    equals = []
    for weights_before, weights_after in zip(model_weights_list_before, model_weights_list_after):
        equals.append(np.array_equal(weights_before, weights_after))
    assert not all(equals)
