"""Tests for the example generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

import os
import shutil

import datetime
import dateutil

from absl.testing import absltest

from goes_truecolor.beam import make_truecolor_examples
from goes_truecolor.tests import fake_gcs
from goes_truecolor.tests import goes_test_util


class MakeTruecolorExamplesTest(absltest.TestCase):
  """Tests for the example generator."""

  def setUp(self):
    super(MakeTruecolorExamplesTest, self).setUp()
    self.tmp_dir = os.path.join(absltest.get_default_test_tmpdir(), 'data')
    if not os.path.exists(self.tmp_dir):
      os.makedirs(self.tmp_dir)
    self.bucket_name = 'test_bucket'
    self.client = fake_gcs.FakeClient(self.tmp_dir)

  def tearDown(self):
    super(MakeTruecolorExamplesTest, self).tearDown()
    shutil.rmtree(self.tmp_dir)

  def create_fake_goes_image(self, t: datetime.datetime, channel: int) -> Text:
    """Create a fake GOES file."""
    dirname = os.path.join(self.tmp_dir, self.bucket_name)
    return goes_test_util.create_fake_goes_image(dirname, t, channel)

  def test_create_tfexamples(self):
    """Test the CreateTFExample class."""
    utc = dateutil.tz.tzutc()
    t = datetime.datetime(2018, 1, 1, 12, 15, 0, tzinfo=utc)
    channels = [1, 2, 3, 7, 8, 9]
    for c in channels:
      self.create_fake_goes_image(t, c)

    image_size = 32
    tile_size = 16
    do_fn = make_truecolor_examples.CreateTFExamples(
        project_id='test',
        goes_bucket_name=self.bucket_name,
        image_size=image_size,
        tile_size=tile_size,
        world_map='file:goes_truecolor/tests/testdata/world_map.jpg',
        ir_channels=[7, 8, 9],
        tmp_dir=self.tmp_dir,
        gcs_client=self.client)
    tiles = list(do_fn.process(t))
    self.assertEqual((image_size // tile_size) ** 2, len(tiles))

  def test_create_tfexamples_with_no_data(self):
    """Test the CreateTFExample class."""
    utc = dateutil.tz.tzutc()
    t = datetime.datetime(2018, 1, 1, 12, 15, 0, tzinfo=utc)

    image_size = 32
    tile_size = 16
    do_fn = make_truecolor_examples.CreateTFExamples(
        project_id='test',
        goes_bucket_name=self.bucket_name,
        image_size=image_size,
        tile_size=tile_size,
        world_map='file:goes_truecolor/tests/testdata/world_map.jpg',
        ir_channels=[7, 8, 9],
        tmp_dir=self.tmp_dir,
        gcs_client=self.client)
    tiles = list(do_fn.process(t))
    self.assertEqual([], tiles)

  def test_create_tfexamples_with_missing_channel(self):
    """Test the CreateTFExample class."""
    utc = dateutil.tz.tzutc()
    t = datetime.datetime(2018, 1, 1, 12, 15, 0, tzinfo=utc)
    channels = [1, 2, 3, 7, 9]
    for c in channels:
      self.create_fake_goes_image(t, c)

    image_size = 32
    tile_size = 16
    do_fn = make_truecolor_examples.CreateTFExamples(
        project_id='test',
        goes_bucket_name=self.bucket_name,
        image_size=image_size,
        tile_size=tile_size,
        world_map='file:goes_truecolor/tests/testdata/world_map.jpg',
        ir_channels=[7, 8, 9],
        tmp_dir=self.tmp_dir,
        gcs_client=self.client)
    tiles = list(do_fn.process(t))
    self.assertEqual((image_size // tile_size) ** 2, len(tiles))


if __name__ == '__main__':
  absltest.main()
