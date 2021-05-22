import numpy as np
import math


def image2tiles(im, h, w):
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


def tiles2images(tiles, im_shape, h, w):
    """Transforms a list of tiles to an image

    Parameters
    ----------
    tiles : list
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


def get_labels_tiles(tiles):
    """Transforms a list of tiles of labels to a one-hot encoded array of labels, based on the minimum distance in the
    color space to a set of colors (white, blue, etc.)

    Parameters
    ----------
    tiles : list
        List of tiles

    Returns
    -------
    numpy array
        One-hot encoded labels
    """
    output_channels = 5
    tiles_len = len(tiles)
    result = np.zeros((tiles_len, tiles[0].shape[0], tiles[0].shape[1], output_channels))
    colours = {
        0: (255, 255, 255),  # White
        1: (0, 0, 255),  # Blue
        2: (0, 255, 255),  # Turquoise
        3: (0, 255, 0),  # Green
        4: (255, 255, 0),  # Yellow
        # 5: (255, 0, 0),  # Red
    }
    for i, tile in enumerate(tiles):
        for colour_index, colour in colours.items():
            r_channel = np.abs(tile[:, :, 0] - colour[2])
            g_channel = np.abs(tile[:, :, 1] - colour[1])
            b_channel = np.abs(tile[:, :, 2] - colour[0])
            color_distance = r_channel + g_channel + b_channel
            result[i, :, :, colour_index] = color_distance
    result = np.argmin(result, axis=-1)  # Min color distance
    result2 = np.zeros((tiles_len, tiles[0].shape[0], tiles[0].shape[1], output_channels))
    for colour_index, colour in colours.items():
        mat = result[:, :, :] == colour_index
        result2[:, :, :, colour_index] = mat
    return result2
