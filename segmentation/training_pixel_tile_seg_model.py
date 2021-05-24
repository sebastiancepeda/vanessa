import pandas as pd
from loguru import logger
import cv2
import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy

from cv.image_processing import image2tiles, get_labels_tiles, predictions2image, tiles2images
from cv.tf_utils import train_model
from segmentation.pixel_tile_segmentation_model import get_model_definition


def get_params():
    path = '/home/sebastian/projects/'
    input_folder = "./test_data/"
    output_folder = f'./output/'
    tile_model_folder = "./tmp/tile_model/"
    tile_model_file = f"{tile_model_folder}/tile_model.save"
    pixel_model_folder = "./tmp/pixel_model/"
    pixel_model_file = f"{pixel_model_folder}/pixel_model.save"
    params = {
        'tile_model_params': {
            'epochs': 10,
            'model_folder': tile_model_folder,
            'model_file': tile_model_file,
        },
        'pixel_model_params': {
            'epochs': 10,
            'model_folder': pixel_model_folder,
            'model_file': pixel_model_file,
        },
        'input_folder': input_folder,
        'output_folder': output_folder,
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
    h = params['h']
    w = params['w']
    path = "/home/sebastian/vaihingen"
    sets = f"{path}/sets.csv"
    sets = pd.read_csv(sets, sep=',')
    sets['image_file'] = path + '/images/' + sets.image.map(str)
    sets['label_file'] = path + '/labels/' + sets.image.map(str)
    train_data_1 = sets.query("set == 'training_1'")
    train_data_tile = sets.query("set == 'training_2'").head(20)
    test_data_tile = sets.query("set == 'training_2'").head(3)
    test_data = sets.query("set == 'test'")
    logger.info(f"Creating models")
    features_model, tile_model, pixel_model = get_model_definition(
        img_height=h, img_width=w, in_channels=3, out_channels=5)
    image_file_train_tile = train_data_tile.image_file
    label_file_train_tile = train_data_tile.label_file
    image_file_test_tile = test_data_tile.image_file
    label_file_test_tile = test_data_tile.label_file
    x_train_tile = [cv2.imread(f) for f in image_file_train_tile]
    y_train_tile = [cv2.imread(f) for f in label_file_train_tile]
    x_test_tile = [cv2.imread(f) for f in image_file_test_tile]
    y_test_tile = [cv2.imread(f) for f in label_file_test_tile]
    x_train_tile = [get_image(f, h, w) for f in x_train_tile]
    y_train_tile = [get_label(f, h, w) for f in y_train_tile]
    x_test_tile = [get_image(f, h, w) for f in x_test_tile]
    y_test_tile = [get_label(f, h, w) for f in y_test_tile]
    x_train_tile = np.concatenate(x_train_tile, axis=0)
    y_train_tile = np.concatenate(y_train_tile, axis=0)
    x_test_tile = np.concatenate(x_test_tile, axis=0)
    y_test_tile = np.concatenate(y_test_tile, axis=0)
    y_train_tile = y_train_tile.any(axis=(1, 2)).astype(np.float)
    y_test_tile = y_test_tile.any(axis=(1, 2)).astype(np.float)
    logger.info(f"Training tile model")
    tile_model.compile(
        optimizer='adam',
        loss="mse",
        metrics=['mae'])
    tile_model = train_model(
        x_train_tile, y_train_tile, x_test_tile, y_test_tile, tile_model, params['tile_model_params'], logger)
    features_model.trainable = False
    image_file_train_1 = train_data_1.image_file
    label_file_train_1 = train_data_1.label_file
    image_file_test = test_data.image_file
    label_file_test = test_data.label_file
    x_train_1 = [cv2.imread(f) for f in image_file_train_1]
    y_train_1 = [cv2.imread(f) for f in label_file_train_1]
    x_test = [cv2.imread(f) for f in image_file_test]
    y_test = [cv2.imread(f) for f in label_file_test]
    x_train_1 = [get_image(f, h, w) for f in x_train_1]
    y_train_1 = [get_label(f, h, w) for f in y_train_1]
    x_test = [get_image(f, h, w) for f in x_test]
    y_test = [get_label(f, h, w) for f in y_test]
    x_train_1 = np.concatenate(x_train_1, axis=0)
    y_train_1 = np.concatenate(y_train_1, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    # y_debug1 = tiles2images(y, im_shape, h, w)
    # cv2.imwrite("tmp/y_debug1.png", y_debug1)
    logger.info(f"Training pixel model")
    pixel_model.compile(
        optimizer='adam',
        loss="mse",
        metrics=['mae'])
    pixel_model = train_model(x_train_1, y_train_1, x_test, y_test, pixel_model, params['pixel_model_params'], logger)
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
