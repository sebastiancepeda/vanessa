import numpy as np
import math

from typing import List


def image2tiles(im: np.ndarray, h: int, w: int):
    """Transforms an image to a list of tiles

    Parameters
    ----------
    im : numpy array
        Image
    h : int
        Height of the tiles
    w : int
        Width of the tiles

    Returns
    -------
    tiles
        List of tiles
    """
    im_height, im_width, channels = im.shape
    n_h = math.ceil(im_height / h)
    n_w = math.ceil(im_width / w)
    tiles = []
    for y_index in range(n_h):
        for x_index in range(n_w):
            x0, x1 = (x_index * w, (x_index + 1) * w)
            y0, y1 = (y_index * h, (y_index + 1) * h)
            delta_x = x1 - im_width
            delta_y = y1 - im_height
            if delta_x > 0:
                x0 = x0 - delta_x
                x1 = x1 - delta_x
            if delta_y > 0:
                y0 = y0 - delta_y
                y1 = y1 - delta_y
            tile = im[y0:y1, x0:x1, :]
            assert tile.shape[0] == h
            assert tile.shape[1] == w
            tiles.append(tile)
    return tiles


def tiles2images(tiles: List[np.ndarray], im_shape: tuple, h: int, w: int):
    """Transforms a list of tiles to an image

    Parameters
    ----------
    tiles : List[np.ndarray]
        List of tiles
    im_shape : tuple
        Shape of the image
    h : int
        Height of the tiles
    w : int
        Width of the tiles

    Returns
    -------
    numpy array
        Image
    """
    im_height, im_width, channels = im_shape
    n_h = math.ceil(im_height / h)
    n_w = math.ceil(im_width / w)
    im = []
    for y_index in range(n_h):
        im_row = tiles[y_index * n_w:(y_index + 1) * n_w]
        dw = im_width % w
        im_row[-1] = im_row[-1][:, -dw:]
        im_row = np.concatenate(im_row, axis=1)
        im.append(im_row)
    dh = im_height % h
    im[-1] = im[-1][-dh:, :]
    im = np.concatenate(im, axis=0)
    return im


def get_color_map():
    """
    Gets color mapping
    Returns
    -------
    dict
        color mapping
    """
    color_map = {
        0: (255, 255, 255),  # White
        1: (0, 0, 255),  # Blue
        2: (0, 255, 255),  # Turquoise
        3: (0, 255, 0),  # Green
        4: (255, 255, 0),  # Yellow
        # 5: (255, 0, 0),  # Red
    }
    return color_map


def get_labels_tiles(tiles: List[np.ndarray]):
    """Transforms a list of tiles of labels to a one-hot encoded array of labels, based on the minimum distance in the
    color space to a set of colors (white, blue, etc.)

    Parameters
    ----------
    tiles : List[np.ndarray]
        List of tiles

    Returns
    -------
    numpy array
        One-hot encoded labels
    """
    output_channels = 5
    tiles_len = len(tiles)
    result = np.zeros((tiles_len, tiles[0].shape[0], tiles[0].shape[1], output_channels))
    color_map = get_color_map()
    for i, tile in enumerate(tiles):
        for colour_index, colour in color_map.items():
            r_channel = np.abs(tile[:, :, 0] - colour[2])
            g_channel = np.abs(tile[:, :, 1] - colour[1])
            b_channel = np.abs(tile[:, :, 2] - colour[0])
            color_distance = r_channel + g_channel + b_channel
            result[i, :, :, colour_index] = color_distance
    result = np.argmin(result, axis=-1)  # Min color distance
    result2 = np.zeros((tiles_len, tiles[0].shape[0], tiles[0].shape[1], output_channels))
    for colour_index, colour in color_map.items():
        mat = result[:, :, :] == colour_index
        result2[:, :, :, colour_index] = mat
    return result2


def predictions2image(y_pred: np.ndarray, im_shape: tuple, h: int, w: int):
    """
    Transforms an numpy array of predictions into an image

    Parameters
    ----------
    y_pred : Numpy array of predictions
    im_shape : Shape of the output image
    h : Height of the image
    w : Width of the image

    Returns
    -------
    np.ndarray:
        Output image
    """
    y_pred2 = np.zeros((y_pred.shape[0], y_pred.shape[1], y_pred.shape[2], 3))
    color_map = get_color_map()
    for colour_index, colour in color_map.items():
        sub_mat = y_pred == colour_index
        r, g, b = colour
        y_pred2[sub_mat, 0] = b
        y_pred2[sub_mat, 1] = g
        y_pred2[sub_mat, 2] = r
    y_pred2 = [y_pred2[i] for i in range(y_pred2.shape[0])]
    y_pred2 = tiles2images(y_pred2, im_shape, h, w)
    return y_pred2
