"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import cv2
import numpy as np

from cv.image_processing import image2tiles, tiles2images


def test_split_and_merge():
    folder = "./test_data/"
    image = "top_mosaic_09cm_area1_image_resized.tif"
    im_file = f"{folder}/{image}"
    im = cv2.imread(im_file)
    assert im is not None
    print(im.shape)
    h = 200
    w = 200
    tiles = image2tiles(im, h, w)
    im2 = tiles2images(tiles, im.shape, h, w)
    # cv2.imwrite("filename.png", im2)
    diff = im - im2
    diff = np.abs(diff).sum()
    assert diff == 0
