import pandas as pd
from loguru import logger
import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam

from cv.image_processing import image2tiles, get_labels_tiles, predictions2image
from cv.tf_utils import train_model
from segmentation.pixel_segmentation_model import get_model_definition as get_pixel_model
from segmentation.pixel_tile_segmentation_model import get_model_definition as get_pixel_tile_model


def get_params():
    path = '/home/sebastian/projects/'
    input_folder = "./test_data/"
    output_folder = f'./output/'
    tile_model_folder = "./tmp/tile_model/"
    tile_model_file = f"{tile_model_folder}/tile_model.save"
    pixel_model_0_folder = "./tmp/pixel_model_0/"
    pixel_model_0_file = f"{pixel_model_0_folder}/pixel_model_0.save"
    pixel_model_1_folder = "./tmp/pixel_model_1/"
    pixel_model_1_file = f"{pixel_model_1_folder}/pixel_model_1.save"
    epochs = 100  # 20
    params = {
        'tile_model_params': {
            'epochs': epochs * 2,
            'model_folder': tile_model_folder,
            'model_file': tile_model_file,
        },
        'pixel_model_0_params': {
            'epochs': epochs,
            'model_folder': pixel_model_0_folder,
            'model_file': pixel_model_0_file,
        },
        'pixel_model_1_params': {
            'epochs': epochs,
            'model_folder': pixel_model_1_folder,
            'model_file': pixel_model_1_file,
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


def get_pixel_data(h, w, train_data, test_data):
    x_train = [cv2.imread(f) for f in train_data.image_file]
    y_train = [cv2.imread(f) for f in train_data.label_file]
    x_test = [cv2.imread(f) for f in test_data.image_file]
    y_test = [cv2.imread(f) for f in test_data.label_file]
    x_train = [get_image(f, h, w) for f in x_train]
    y_train = [get_label(f, h, w) for f in y_train]
    x_test = [get_image(f, h, w) for f in x_test]
    y_test = [get_label(f, h, w) for f in y_test]
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    return x_train, y_train, x_test, y_test


def get_tile_data(h, w, train_data, test_data):
    x_train, y_train, x_test, y_test = get_pixel_data(h, w, train_data, test_data)
    y_train = y_train.any(axis=(1, 2)).astype(np.float64)
    y_test = y_test.any(axis=(1, 2)).astype(np.float64)
    # y_train = y_train.mean(axis=(1, 2)).astype(np.float64)
    # y_test = y_test.mean(axis=(1, 2)).astype(np.float64)
    return x_train, y_train, x_test, y_test


def save_inference(file_name, pixel_model, h, w, prefix):
    logger.info(f"Getting inference for [{file_name}]")
    x = cv2.imread(file_name)
    im_shape = x.shape
    x = get_image(x, h, w)
    y_pred = get_prediction(pixel_model, x, im_shape, h, w)
    f_name = file_name.split('/')[-1].split('.')[0]
    f_name = f"output/{f_name}_{prefix}_pred.png"
    cv2.imwrite(f_name, y_pred)


def train_pixel_tile_seg_model(params):
    logger.add("logs/train_pixel_tile_seg_model_{time}.log")
    logger.info(f"Reading parameters")
    h = params['h']
    w = params['w']
    dim = 8
    pixel_0_params = params['pixel_model_0_params']
    pixel_1_params = params['pixel_model_1_params']
    tile_params = params['tile_model_params']
    path = "/home/sebastian/vaihingen"
    sets = f"{path}/sets.csv"
    logger.info(f"Reading datasets location")
    sets = pd.read_csv(sets, sep=',')
    sets['image_file'] = path + '/images/' + sets.image.map(str)
    sets['label_file'] = path + '/labels/' + sets.image.map(str)
    # Data sets
    train_data_pixel = sets.query("set == 'training_1'")  # .head(3)
    train_data_tile = sets.query("set == 'training_2'").head(20)  # .head(3)
    test_data_tile = sets.query("set == 'training_2'").tail(3)  # .head(3)
    test_data_pixel = sets.query("set == 'test'")  # .head(3)
    logger.info(f"Loading data")
    x_train_tile, y_train_tile, x_test_tile, y_test_tile = get_tile_data(h, w, train_data_tile, test_data_tile)
    x_train_pixel, y_train_pixel, x_test, y_test = get_pixel_data(h, w, train_data_pixel, test_data_pixel)
    # Model definition
    logger.info(f"Creating models")
    pixel_model_0 = get_pixel_model(img_height=h, img_width=w, in_channels=3, out_channels=5, dim=dim)
    tile_model, pixel_model_1 = get_pixel_tile_model(h, w, in_channels=3, out_channels=5, dim=dim)
    pixel_model_0_w_a = pixel_model_0.get_weights().copy()
    pixel_model_1_w_a = pixel_model_1.get_weights().copy()
    assert not np.array_equal(pixel_model_0_w_a[0], pixel_model_1_w_a[0])  # Different pixel level models
    # All models based on the same U-net like model
    assert np.array_equal(tile_model.get_weights().copy()[0], pixel_model_1.get_weights().copy()[0])
    logger.info(f"Training pixel without weak supervision")
    pixel_model_0 = train_model(x_train_pixel, y_train_pixel, x_test, y_test, pixel_model_0, pixel_0_params, logger)
    pixel_model_0_w_b = pixel_model_0.get_weights().copy()
    logger.info(f"error_pixel_0_train: {np.abs(y_train_pixel - pixel_model_0.predict(x_train_pixel)).mean()}")
    logger.info(f"error_pixel_0_test: {np.abs(y_test - pixel_model_0.predict(x_test)).mean()}")
    assert not np.array_equal(pixel_model_0_w_a[0], pixel_model_0_w_b[0])  # Improvements: training of tile model
    assert not np.array_equal(pixel_model_0.get_weights().copy()[0],
                              pixel_model_1.get_weights().copy()[0])  # Different pixel level models
    for file_name in test_data_pixel.image_file:
        save_inference(file_name, pixel_model_0, h, w, 'pixel_model_0')
    logger.info(f"Training tile model")
    tile_model.compile(optimizer='adam', loss="mse", metrics=['mae'])
    tile_model = train_model(x_train_tile, y_train_tile, x_test_tile, y_test_tile, tile_model, tile_params, logger)
    logger.info(f"error_tile_train: {np.abs(y_train_tile - tile_model.predict(x_train_tile)).mean()}")
    logger.info(f"error_tile_test: {np.abs(y_test_tile - tile_model.predict(x_test_tile)).mean()}")
    error_tile = np.abs(y_test_tile - tile_model.predict(x_test_tile)).mean()
    logger.info(f"error_tile: {error_tile}")
    tile_model.trainable = False
    logger.info(f"Training pixel model (transfer learning)")
    pixel_model_1.compile(optimizer='adam', loss="mse", metrics=['mae'])
    pixel_model_1 = train_model(x_train_pixel, y_train_pixel, x_test, y_test, pixel_model_1, pixel_1_params, logger)
    pixel_model_1_w_b = pixel_model_1.get_weights().copy()
    logger.info(f"error_pixel_1_train: {np.abs(y_train_pixel - pixel_model_1.predict(x_train_pixel)).mean()}")
    logger.info(f"error_pixel_1_test: {np.abs(y_test - pixel_model_1.predict(x_test)).mean()}")
    assert not np.array_equal(pixel_model_1_w_a[0], pixel_model_1_w_b[0])  # Improvements: training of tile model
    assert not np.array_equal(pixel_model_0_w_b[0], pixel_model_1_w_b[0])  # Different pixel level models
    # All models based on the same U-net like model
    assert np.array_equal(tile_model.get_weights().copy()[0], pixel_model_1.get_weights().copy()[0])
    for file_name in test_data_pixel.image_file:
        save_inference(file_name, pixel_model_1, h, w, 'pixel_model_1_tl')


if __name__ == "__main__":
    train_pixel_tile_seg_model(get_params())
