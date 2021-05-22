"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import cv2
import numpy as np
import pytest

from cv.image_processing import image2tiles, tiles2images, get_labels_tiles


def test_split_and_merge():
    folder = "./test_data/"
    image = "top_mosaic_09cm_area1_image_resized.tif"
    im_file = f"{folder}/{image}"
    im = cv2.imread(im_file)
    assert im is not None
    h = 200
    w = 200
    tiles = image2tiles(im, h, w)
    im2 = tiles2images(tiles, im.shape, h, w)
    # cv2.imwrite("filename.png", im2)
    diff = im - im2
    diff = np.abs(diff).sum()
    assert diff == 0


def test_get_labels_tile():
    folder = "./test_data/"
    image = "top_mosaic_09cm_area1_label_resized.tif"
    im_file = f"{folder}/{image}"
    im = cv2.imread(im_file)
    assert im is not None
    h = 200
    w = 200
    tiles = image2tiles(im, h, w)
    tiles_len = len(tiles)
    tiles = get_labels_tiles(tiles)
    assert tiles.min() >= 0
    assert tiles.max() <= 1
    assert tiles.shape[0] == tiles_len
    assert tiles.shape[1] == 200
    assert tiles.shape[2] == 200
    avg = tiles.mean(axis=(0, 1, 2))
    fraction_pixels_assigned = avg.sum()
    assert 1.0 == pytest.approx(fraction_pixels_assigned)  # Almost all pixels assigned
