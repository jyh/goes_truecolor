"""Tests for the example generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

import io
import os
import shutil

import datetime
import dateutil
import numpy as np
from PIL import Image

from absl.testing import absltest

from goes_truecolor.beam import make_cloud_masks
from goes_truecolor.lib import goes_reader
from goes_truecolor.tests import fake_datastore
from goes_truecolor.tests import fake_gcs
from goes_truecolor.tests import goes_test_util


class MakeCloudMasksTest(absltest.TestCase):
  """Tests for the example generator."""

  def setUp(self):
    super(MakeCloudMasksTest, self).setUp()
    self.tmp_dir = os.path.join(absltest.get_default_test_tmpdir(), 'data')
    if not os.path.exists(self.tmp_dir):
      os.makedirs(self.tmp_dir)
    self.bucket_name = 'test_bucket'
    self.gcs_client = fake_gcs.FakeClient(self.tmp_dir)
    self.ds_client = fake_datastore.FakeClient()

  def tearDown(self):
    super(MakeCloudMasksTest, self).tearDown()
    shutil.rmtree(self.tmp_dir)

  def create_fake_goes_image(self, t: datetime.datetime, channel: int) -> Text:
    """Create a fake GOES file."""
    dirname = os.path.join(self.tmp_dir, self.bucket_name)
    return goes_test_util.create_fake_goes_image(dirname, t, channel)

  def test_create_cloud_masks(self):
    """Test the CreateCloudMasks.process method."""
    utc = dateutil.tz.tzutc()
    t = datetime.datetime(2018, 1, 1, 12, 15, 0, tzinfo=utc)
    files = {}
    for c in goes_reader.IR_CHANNELS:
      files[c] = self.create_fake_goes_image(t, c)

    image_size = 128
    creator = make_cloud_masks.CreateCloudMasks(
        project_id='abc',
        goes_bucket_name=self.bucket_name,
        image_size=image_size,
        tile_size=64,
        tile_border_size=8,
        model_dir='goes_truecolor/tests/testdata/model',
        ir_channels=goes_reader.IR_CHANNELS,
        tmp_dir=self.tmp_dir,
        gcs_client=self.gcs_client,
        ds_client=self.ds_client)
    [entity] = list(creator.process((t, files)))
    self.assertEqual('2018-01-01T12:15:00+00:00', entity.key.val)

    buf = io.BytesIO(entity['mask_jpeg'])
    img = Image.open(buf)
    img = np.array(img)
    self.assertEqual((128, 128), img.shape)
    gip = entity['goes_imager_projection']
    self.assertAlmostEqual(-75, gip['longitude_of_projection_origin'])


if __name__ == '__main__':
  absltest.main()
