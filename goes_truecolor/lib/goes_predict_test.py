"""Test the GoesReader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import numpy as np

from goes_truecolor.lib import goes_predict
from goes_truecolor.lib import goes_reader

TILE_SIZE = 64


class GoesPredictTest(absltest.TestCase):
  """Test GoesPredict."""

  def test_overlapping_tiles(self):  # pylint: disable=no-self-use
    """Test goes_predict._get_overlapping_tiles"""
    x = np.linspace(1, 16, 16)
    img = np.stack(np.meshgrid(x, x), axis=-1)
    cols = goes_predict._get_overlapping_tiles(img, 4, 0)
    for j, rows in enumerate(cols):
      for i, tile in enumerate(rows):
        y = i * 4
        x = j * 4
        np.testing.assert_almost_equal(tile, img[y:y+4, x:x+4, :], err_msg='y={} x={}'.format(y, x))

  def test_overlapping_tiles_with_border(self):  # pylint: disable=no-self-use
    """Test goes_predict._get_overlapping_tiles"""
    x = np.linspace(1, 16, 16)
    img = np.stack(np.meshgrid(x, x), axis=-1)
    cols = goes_predict._get_overlapping_tiles(img, 4, 1)
    img_pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), 'constant')
    for j, rows in enumerate(cols):
      for i, tile in enumerate(rows):
        y = i * 2
        x = j * 2
        np.testing.assert_almost_equal(
            tile, img_pad[y:y+4, x:x+4, :], err_msg='y={} x={}'.format(y, x))

  def test_predict(self):  # pylint: disable=no-self-use
    model = goes_predict.GoesPredict(
        'goes_truecolor/tests/testdata/model', tile_size=TILE_SIZE, tile_border_size=8)
    ir_img = np.zeros((512, 512, len(goes_reader.IR_CHANNELS)), dtype=np.uint8)
    cloud_img = model.predict(ir_img)
    self.assertEqual((512, 512), cloud_img.shape)


if __name__ == '__main__':
  absltest.main()
