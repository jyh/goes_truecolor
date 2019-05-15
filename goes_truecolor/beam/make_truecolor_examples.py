"""Generate Tensorflow training examples from satellite images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Generator, List, Optional, Text

import datetime
import logging  # pylint: disable=unused-import
import os
from absl import app
from absl import flags

import apache_beam as beam
import dateparser
import dateutil
import numpy as np  # pylint: disable=unused-import
import tensorflow as tf  # pylint: disable=unused-import

import google.cloud.storage as gcs

from goes_truecolor.lib import goes_reader  # pylint: disable=unused-import
from goes_truecolor.learning import hparams  # pylint: disable=unused-import


flags.DEFINE_string(
    'project', 'weather-324',
    'Name of the project for billing')

flags.DEFINE_string(
    'goes_bucket', 'gcp-public-data-goes-16',
    'GOES bucket')

flags.DEFINE_string(
    'tmp_dir', 'gs://weather-tmp',
    'Temporary files bucket')

flags.DEFINE_string(
    'out_dir', 'gs://weather-datasets/goes_truecolor/examples',
    'Output bucket')

flags.DEFINE_string(
    'runner', 'DirectRunner',
    'Which beam runner to use; '
    'use DataflowRunner to run on cloud')

flags.DEFINE_integer(
    'max_workers', 10,
    'Maximum number of worker processes')

flags.DEFINE_integer(
    'image_size', 1024, 'size of the images (images are square)')

flags.DEFINE_integer(
    'tile_size', 64, 'size of tensorflow example tiles')

flags.DEFINE_string(
    'world_map',
    'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/'
    'world.topo.bathy.200412.3x5400x2700.jpg',
    'Base map image of the world')

flags.DEFINE_string(
    'train_start_date', '1/1/2018 17:00',
    'Start date for collecting images')

flags.DEFINE_string(
    'train_end_date', '12/31/2018 17:00',
    'End date for collecting images (inclusive)')

flags.DEFINE_integer(
    'train_step_days', 2,
    'Number of days between training samples')

flags.DEFINE_string(
    'test_start_date', '1/1/2019 17:00',
    'Start date for collecting images')

flags.DEFINE_string(
    'test_end_date', '5/1/2019 17:00',
    'End date for collecting images (inclusive)')

flags.DEFINE_integer(
    'test_step_days', 3,
    'Number of days between testing samples')

flags.DEFINE_integer(
    'num_shards', 32, 'number of tfrecord shards')


FLAGS = flags.FLAGS


def _get_sample_dates(start_date: Text, end_date: Text, step_days: int) -> List[datetime.datetime]:
  utc = dateutil.tz.tzutc()
  start_date = dateparser.parse(start_date)
  start_date = start_date.replace(tzinfo=utc)
  end_date = dateparser.parse(end_date)
  end_date = end_date.replace(tzinfo=utc)
  dates = []
  t = start_date
  dt = datetime.timedelta(days=step_days)
  while t <= end_date:
    dates.append(t)
    t += dt
  return dates


# pylint: disable=abstract-method,too-many-instance-attributes
class CreateTFExamples(beam.DoFn):
  """Create tensorflow example protos from GOES images."""

  def __init__(
      self,
      project_id: Text, goes_bucket_name: Text,
      image_size: int, tile_size: int, world_map: Text, ir_channels: List[int],
      tmp_dir: Optional[Text] = None,
      gcs_client: Optional[gcs.Client] = None):
    # pylint: disable=super-init-not-called
    """Read a GOES snapshot and contruct TFExamples.

    Args:
      t: the snapshot time
      project_id: the billing account
      goes_bucket_name: the name of the bucket containg the GOES images
      image_size: the total image size (square)
      tile_size: the size of each example (square)
      world_map: the URL of the world map image
      ir_channels: the IR channels.
    """
    self.project_id = project_id
    self.goes_bucket_name = goes_bucket_name
    self.image_size = image_size
    self.tile_size = tile_size
    self.world_map = world_map
    self.ir_channels = ir_channels
    self.reader = None
    self.tmp_dir = tmp_dir
    self.gcs_client = gcs_client

  # pylint: disable=arguments-differ,too-many-locals
  def process(self, t: datetime.datetime) -> Generator[Text, None, None]:
    # pylint: disable=reimported,redefined-outer-name
    import logging
    import numpy as np
    import tensorflow as tf
    from goes_truecolor.lib import goes_reader
    from goes_truecolor.learning import hparams

    # Create the GoesReader lazily so that beam will not pickle it
    # when copying this object to other workers.
    if self.reader is None:
      logging.info('creating GoesReader')
      shape = self.image_size, self.image_size
      self.reader = goes_reader.GoesReader(
          project_id=self.project_id,
          goes_bucket_name=self.goes_bucket_name, shape=shape,
          tmp_dir=self.tmp_dir, client=self.gcs_client)

    # Fetch the truecolor and IR images.
    logging.info('creating cloud mask image %s', t)
    mask_img = self.reader.cloud_mask(t)
    if mask_img is None:
      return
    logging.info('creating IR image for %s', t)
    ir, _ = self.reader.raw_image(t, self.ir_channels)
    if ir is None:
      return

    # Split into tiles and generate tensorflow examples.
    logging.info('creating tiles for %s', t)
    partitions = self.image_size // self.tile_size
    mask_rows = np.split(mask_img, partitions, axis=0)
    ir_rows = np.split(ir, partitions, axis=0)
    for mask_row, ir_row in zip(mask_rows, ir_rows):
      mask_tiles = np.split(mask_row, partitions, axis=1)
      ir_tiles = np.split(ir_row, partitions, axis=1)
      for mask_tile, ir_tile in zip(mask_tiles, ir_tiles):
        features = {
            hparams.CLOUD_MASK_FEATURE_NAME: tf.train.Feature(
                int64_list=tf.train.Int64List(value=mask_tile.ravel())),

            hparams.IR_CHANNELS_FEATURE_NAME:  tf.train.Feature(
                int64_list=tf.train.Int64List(value=ir_tile.ravel())),
        }
        example = tf.train.Example(features=tf.train.Features(feature=features))
        yield example.SerializeToString()


def main(unused_argv):
  """Beam pipeline to create examples."""
  train_dates = _get_sample_dates(
      FLAGS.train_start_date, FLAGS.train_end_date, FLAGS.train_step_days)
  test_dates = _get_sample_dates(
      FLAGS.test_start_date, FLAGS.test_end_date, FLAGS.test_step_days)

  # Create the beam pipeline.
  options = {'staging_location': os.path.join(FLAGS.tmp_dir, 'tmp', 'staging'),
             'temp_location': os.path.join(FLAGS.tmp_dir, 'tmp'),
             'job_name': datetime.datetime.now().strftime('truecolor-%y%m%d-%H%M%S'),
             'project': FLAGS.project,
             'num_workers': 1,
             'max_num_workers': FLAGS.max_workers,
             'machine_type': 'n1-highmem-4',
             'setup_file': os.path.join(
                 os.path.dirname(os.path.abspath(__file__)), '../../setup.py'),
             'teardown_policy': 'TEARDOWN_ALWAYS',
             'save_main_session': False}
  opts = beam.pipeline.PipelineOptions(flags=[], **options)

  # Run the beam pipeline.
  with beam.Pipeline(FLAGS.runner, options=opts) as p:
    for mode, dates in [('train', train_dates), ('test', test_dates)]:
      (p  # pylint: disable=expression-not-assigned
       | 'create-{}'.format(mode) >> beam.Create(dates)
       | 'sample-{}'.format(mode) >> beam.ParDo(CreateTFExamples(
           FLAGS.project, FLAGS.goes_bucket,
           FLAGS.image_size, FLAGS.tile_size, FLAGS.world_map,
           goes_reader.IR_CHANNELS))
       | 'write-{}'.format(mode) >> beam.io.tfrecordio.WriteToTFRecord(
           os.path.join(FLAGS.out_dir, '{}.tfrecord'.format(mode)),
           num_shards=FLAGS.num_shards))


if __name__ == '__main__':
  app.run(main)
