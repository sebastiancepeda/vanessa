import pandas as pd
from loguru import logger
import cv2
import numpy as np

from cv.image_processing import image2tiles, get_labels_tiles, predictions2image, tiles2images
from cv.tf_utils import train_model
from segmentation.pixel_tile_segmentation_model import get_model_definition


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
        'epochs': 100,
        'h': 200,
        'w': 200
    }
    return params


def get_image(image, h, w):
    assert image is not None
    image = image2tiles(image, h, w)
    image = [t[np.newaxis, ...] for t in image]
    image = np.concatenate(image, axis=0)
    return image


def get_label(labels, h, w):
    assert labels is not None
    labels = image2tiles(labels, h, w)
    labels = get_labels_tiles(labels)
    return labels


def get_prediction(model, x, im_shape, h, w):
    y_pred = model.predict(x).round()
    y_pred = predictions2image(y_pred, im_shape, h, w)
    return y_pred


def train_pixel_tile_seg_model(params):
    path = "/home/sebastian/vaihingen"
    sets = f"{path}/sets.csv"
    sets = pd.read_csv(sets, sep=',')
    sets['image_file'] = path + '/images/' + sets.image.map(str)
    sets['label_file'] = path + '/labels/' + sets.image.map(str)
    train_data = sets.query("set == 'training_1'")
    test_data = sets.query("set == 'test'")
    image_file_train = train_data.image_file
    label_file_train = train_data.label_file
    image_file_test = test_data.image_file
    label_file_test = test_data.label_file
    h = params['h']
    w = params['w']
    x_train = [cv2.imread(f) for f in image_file_train]
    y_train = [cv2.imread(f) for f in label_file_train]
    x_test = [cv2.imread(f) for f in image_file_test]
    y_test = [cv2.imread(f) for f in label_file_test]
    x_train = [get_image(f, h, w) for f in x_train]
    y_train = [get_label(f, h, w) for f in y_train]
    x_test = [get_image(f, h, w) for f in x_test]
    y_test = [get_label(f, h, w) for f in y_test]
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    # y_debug1 = tiles2images(y, im_shape, h, w)
    # cv2.imwrite("tmp/y_debug1.png", y_debug1)
    features_model, tile_model, pixel_model = get_model_definition(
        img_height=h, img_width=w, in_channels=3, out_channels=5)
    pixel_model = train_model(x_train, y_train, x_test, y_test, pixel_model, params, logger)
    for file_name in image_file_test:
        logger.info(f"Getting inference for [{file_name}]")
        x = cv2.imread(file_name)
        im_shape = x.shape
        x = get_image(x, h, w)
        y_pred = get_prediction(pixel_model, x, im_shape, h, w)
        f_name = file_name.split('/')[-1].split('.')[0]
        f_name = f"output/{f_name}_pred.png"
        cv2.imwrite(f_name, y_pred)
        # y_debug = predictions2image(y, im_shape, h, w)
        # cv2.imwrite("tmp/y_debug.png", y_debug)


if __name__ == "__main__":
    train_pixel_tile_seg_model(get_params())
