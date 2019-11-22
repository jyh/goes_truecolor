"""Reader for GOES satellite data on Google Cloud Storage."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from typing import List, Text

import numpy as np
import tensorflow.compat.v1 as tf


def _get_overlapping_tiles(
    img: np.ndarray, tile_size: int, tile_border_size: int) -> List[List[np.ndarray]]:
  """Split an image into tiles.

  Args:
    img: the image, of shape (height, width, channels).
    tile_size: the size of square
    tile_border_size: a border around each tile

  Returns:
    A list of columns, each of which is a row of tiles.
  """
  h, _, _ = img.shape
  inner_tile_size = tile_size - 2 * tile_border_size
  num_tiles = (h + inner_tile_size - 1) // inner_tile_size
  new_h = num_tiles * inner_tile_size + 2 * tile_border_size
  pad_l = tile_border_size
  pad_r = new_h - tile_border_size - h
  img = np.pad(img, ((pad_l, pad_r), (pad_l, pad_r), (0, 0)), 'constant')
  cols = []
  for x in range(0, new_h - tile_size + 1, inner_tile_size):
    rows = []
    for y in range(0, new_h - tile_size + 1, inner_tile_size):
      rows.append(img[y:y+tile_size, x:x+tile_size, :])
    cols.append(rows)
  return cols


class GoesPredict():  # pylint: disable=too-few-public-methods
  """Predict a value using a saved Keras model."""

  def __init__(self, model_dir: Text, tile_size: int, tile_border_size: int):
    """Create a predictor.

    Args:
      model_dir: the directory containing the Keras model
      tile_size: tile size expected by the model
      tile_border_size: tile border to crop
    """
    self.model_dir = model_dir
    self.tile_size = tile_size
    self.tile_border_size = tile_border_size

    # Load the model.
    dirnames = tf.gfile.Glob(os.path.join(model_dir, '[0-9]*'))
    model_file = sorted(dirnames)[-1]
    tf.logging.info('Loading model from %s', model_file)
    self.model = tf.keras.experimental.load_from_saved_model(model_file)
    self.model.summary()

  def predict(self, raster_in_img: np.ndarray) -> np.ndarray:
    """Apply the model to make a prediction.

    Args:
      raster_in_img: the input image, dtype=np.uint8.

    Return:
      The prediction, dtype=np.uint8.
    """
    raster_in_img = raster_in_img.astype(np.float32) / 256
    h, w, _ = raster_in_img.shape
    raster_in_tiles = _get_overlapping_tiles(
        raster_in_img, self.tile_size, self.tile_border_size)

    # Assemble the prediction.
    predict_cols = []
    border = self.tile_border_size
    for raster_in_col in raster_in_tiles:
      raster_in_col = np.array(raster_in_col)
      predict_col = self.model.predict(raster_in_col)
      tiles = [predict_col[b, border:-border, border:-border, :]
               for b in range(predict_col.shape[0])]
      predict_col = np.concatenate(tiles, axis=0)
      predict_cols.append(predict_col)
    predict_img = np.concatenate(predict_cols, axis=1)
    predict_img = predict_img[:h, :w, 0]
    predict_img = (predict_img * 255.9).astype(np.uint8)
    return predict_img
