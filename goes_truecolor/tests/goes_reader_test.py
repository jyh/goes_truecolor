"""Test the GoesReader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

import os
import shutil

import datetime
import dateutil

from absl.testing import absltest

from goes_truecolor.lib import goes_reader
from goes_truecolor.tests import fake_gcs
from goes_truecolor.tests import goes_test_util

WORLD_MAP = 'file:goes_truecolor/tests/testdata/world_map.jpg'


class GoesReaderTest(absltest.TestCase):
  """Test the GoesReader."""

  def setUp(self):
    super(GoesReaderTest, self).setUp()
    self.tmp_dir = os.path.join(absltest.get_default_test_tmpdir(), 'data')
    if not os.path.exists(self.tmp_dir):
      os.makedirs(self.tmp_dir)
    self.client = fake_gcs.FakeClient(self.tmp_dir)


  def tearDown(self):
    super(GoesReaderTest, self).tearDown()
    shutil.rmtree(self.tmp_dir)

  def create_fake_goes_image(self, t: datetime.datetime, channel: int) -> Text:
    """Create a fake GOES image."""
    dirname = os.path.join(self.tmp_dir, goes_reader.GOES_BUCKET)
    return goes_test_util.create_fake_goes_image(dirname, t, channel)

  def test_parse_filename(self):
    """Test goes_reader._parse_filename."""
    utc = dateutil.tz.tzutc()
    t = datetime.datetime(2018, 1, 1, 0, 15, 0, tzinfo=utc)
    channel = 1
    filename = goes_test_util.goes_filename(t, channel)
    goes_file = goes_reader._parse_filename(filename)  # pylint: disable=protected-access
    self.assertEqual(filename, goes_file.path)
    self.assertEqual(channel, goes_file.channel)
    self.assertEqual(t, goes_file.start_date)
    self.assertEqual(t, goes_file.end_date)
    self.assertEqual(t, goes_file.creation_date)
    self.assertEqual('ABI-L1b-RadF', goes_file.product)
    self.assertEqual(3, goes_file.mode)

  def test_list_time_range(self):
    """Test GoesReader.list_time_range."""
    utc = dateutil.tz.tzutc()
    t = datetime.datetime(2018, 1, 1, 12, 15, 0, tzinfo=utc)
    channel = 1
    filename = self.create_fake_goes_image(t, channel)

    reader = goes_reader.GoesReader(
        project_id='test',
        shape=(4, 4),
        tmp_dir=self.tmp_dir,
        client=self.client)
    start_time = t - datetime.timedelta(hours=1)
    end_time = t + datetime.timedelta(hours=1)
    [(actual_t, d)] = reader.list_time_range(start_time, end_time)
    self.assertEqual(t, actual_t)
    self.assertEqual(filename, d[1].id)

  def test_load_channel_images(self):
    """Test GoesReader.load_channel_images."""
    utc = dateutil.tz.tzutc()
    t = datetime.datetime(2018, 1, 1, 12, 15, 0, tzinfo=utc)
    channel = 1
    self.create_fake_goes_image(t, channel)

    reader = goes_reader.GoesReader(
        project_id='test',
        shape=(4, 4),
        tmp_dir=self.tmp_dir,
        client=self.client)
    table = reader.load_channel_images(t, [1])
    img, md = table[1]
    self.assertEqual((4, 4), img.shape)
    self.assertAlmostEqual(1e-2, md['kappa0'])

  def test_cloud_mask(self):
    """Test GoesReader.cloud_mask."""
    utc = dateutil.tz.tzutc()
    t = datetime.datetime(2018, 1, 1, 12, 15, 0, tzinfo=utc)
    channels = [1]
    for c in channels:
      self.create_fake_goes_image(t, c)
    reader = goes_reader.GoesReader(
        project_id='test',
        shape=(4, 4),
        tmp_dir=self.tmp_dir,
        client=self.client)
    mask = reader.cloud_mask(t)
    self.assertEqual((4, 4), mask.shape)

  def test_raw_image(self):
    """Test GoesReader.raw_image."""
    utc = dateutil.tz.tzutc()
    t = datetime.datetime(2018, 1, 1, 12, 15, 0, tzinfo=utc)
    channels = [7, 8, 9, 10]
    for c in channels:
      self.create_fake_goes_image(t, c)
    reader = goes_reader.GoesReader(
        project_id='test',
        shape=(4, 4),
        tmp_dir=self.tmp_dir,
        client=self.client)
    img = reader.raw_image(t, channels)
    self.assertEqual((4, 4, 4), img.shape)


if __name__ == '__main__':
  absltest.main()
