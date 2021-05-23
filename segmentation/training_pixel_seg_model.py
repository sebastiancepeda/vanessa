from loguru import logger
import cv2
import numpy as np

from cv.image_processing import image2tiles, get_labels_tiles, predictions2image
from cv.tf_utils import train_model
from segmentation.pixel_segmentation_model import get_model_definition


def get_params():
    path = '/home/sebastian/projects/'
    model_folder = "./tmp/"
    model_file = f"{model_folder}/model"
    input_folder = "./test_data/"
    output_folder = f'./output/'
    params = {
        'model_folder': model_folder,
        'model_file': model_file,
        'input_folder': input_folder,
        'output_folder': output_folder,
        'epochs': 20,
        'h': 200,
        'w': 200
    }
    return params


def train_pixel_seg_model(params):
    x = "top_mosaic_09cm_area1_image_resized.tif"
    x = f"{params['input_folder']}/{x}"
    x = cv2.imread(x)
    im_shape = x.shape
    assert x is not None
    h = params['h']
    w = params['w']
    x = image2tiles(x, h, w)
    x = [t[np.newaxis, ...] for t in x]
    x = np.concatenate(x, axis=0)
    y = "top_mosaic_09cm_area1_label_resized.tif"
    y = f"{params['input_folder']}/{y}"
    y = cv2.imread(y)
    assert y is not None
    y = image2tiles(y, h, w)
    y = get_labels_tiles(y)
    model = get_model_definition(img_height=h, img_width=w, in_channels=3, out_channels=5)
    y_pred = model.predict(x).round()
    logger.info(f"y_pred.mean(): {y_pred.mean()}")
    model = train_model(x, y, x, y, model, params, logger)
    y_pred = model.predict(x).round()
    logger.info(f"y_pred.mean(): {y_pred.mean()}")
    y_pred = np.argmax(y_pred, axis=-1)
    logger.info(f"y_pred.mean(): {y_pred.mean()}")
    y_pred = predictions2image(y_pred, im_shape, h, w)
    logger.info(f"y_pred.mean(): {y_pred.mean()}")
    cv2.imwrite("y_pred.png", y_pred)


if __name__ == "__main__":
    train_pixel_seg_model(get_params())
