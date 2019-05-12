"""Generate Tensorflow training examples from satellite images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Generator, List, Text

import datetime
import logging
import os
from absl import app
from absl import flags

import apache_beam as beam
import dateparser
import dateutil
import numpy as np
import tensorflow as tf

from truecolor.goeslib import goes_reader
from truecolor.learning import hparams


flags.DEFINE_string(
    'project', 'weather-324',
    'Name of the project for billing')

flags.DEFINE_string(
    'goes_bucket', 'gcp-public-data-goes-16',
    'GOES bucket')

flags.DEFINE_string(
    'out_dir', 'gs://weather-tmp/examples',
    'Output bucket')

flags.DEFINE_string(
    'runner', 'DirectRunner',
    'Which beam runner to use; '
    'use DataflowRunner to run on cloud')

flags.DEFINE_integer(
    'max_workers', 10,
    'Maximum number of worker processes')

flags.DEFINE_integer(
    'image_size', 1024, 'size of the images')

flags.DEFINE_string(
    'world_map',
    'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/'
    'world.topo.bathy.200412.3x5400x2700.jpg',
    'Base map image of the world')

flags.DEFINE_string(
    'train_start_date', '1/1/2019 17:00',
    'Start date for collecting images')

flags.DEFINE_string(
    'train_end_date', '1/1/2019 17:00',
    'End date for collecting images (inclusive)')

flags.DEFINE_integer(
    'train_step_days', 1,
    'Number of days between training samples')

flags.DEFINE_string(
    'test_start_date', '2/1/2019 17:00',
    'Start date for collecting images')

flags.DEFINE_string(
    'test_end_date', '2/1/2019 17:00',
    'End date for collecting images (inclusive)')

flags.DEFINE_integer(
    'test_step_days', 1,
    'Number of days between testing samples')

flags.DEFINE_integer(
    'tile_size', 64, 'size of tensorflow example tiles')


FLAGS = flags.FLAGS

IR_CHANNELS = list(range(7, 17))


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


# pylint: disable=too-many-locals
def make_truecolor_examples(
    t: datetime.datetime, project_id: Text, goes_bucket_name: Text,
    image_size: int, tile_size: int, world_map: Text, ir_channels: List[int]) -> Generator[bytes, None, None]:
  """Read a GOES snapshot and contruct TFExamples.

  Args:
    t: the snapshot time
    project_id: the billing account
    goes_bucket_name: the name of the bucket containg the GOES images
    image_size: the total image size (square)
    tile_size: the size of each example (square)
    world_map: the URL of the world map image

  Yields:
    A sequence of serialized TFExample protos.
  """
  # pylint: disable=reimported,redefined-outer-name
  import logging
  import numpy as np
  import tensorflow as tf
  from truecolor.goeslib import goes_reader
  from truecolor.learning import hparams

  # Fetch the truecolor and IR images.
  shape = image_size, image_size
  reader = goes_reader.GoesReader(project_id, goes_bucket_name, shape=shape)
  logging.info('creating truecolor image for %s', t)
  world, rgb = reader.truecolor_image(world_map, t)
  logging.info('creating IR image for %s', t)
  ir = reader.raw_image(t, ir_channels)
  ir = np.concatenate((world, ir), axis=-1)

  # Split into tiles and generate tensorflow examples.
  logging.info('creating tiles for %s', t)
  rgb_rows = np.split(rgb, tile_size, axis=0)
  ir_rows = np.split(ir, tile_size, axis=0)
  for rgb_row, ir_row in zip(rgb_rows, ir_rows):
    rgb_tiles = np.split(rgb_row, tile_size, axis=1)
    ir_tiles = np.split(ir_row, tile_size, axis=1)
    for rgb_tile, ir_tile in zip(rgb_tiles, ir_tiles):
      features = {
        hparams.TRUECOLOR_CHANNELS_FEATURE_NAME: tf.train.Feature(
          int64_list=tf.train.Int64List(value=rgb_tile.ravel())),

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
  options = {'staging_location': os.path.join(FLAGS.out_dir, 'tmp', 'staging'),
             'temp_location': os.path.join(FLAGS.out_dir, 'tmp'),
             'job_name': datetime.datetime.now().strftime('truecolor-%y%m%d-%H%M%S'),
             'project': FLAGS.project,
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
       | 'sample-{}'.format(mode) >> beam.FlatMap(
         make_truecolor_examples,
         FLAGS.project, FLAGS.goes_bucket,
         FLAGS.image_size, FLAGS.tile_size, FLAGS.world_map,
         IR_CHANNELS)
       | 'write-{}'.format(mode) >> beam.io.tfrecordio.WriteToTFRecord(
         os.path.join(FLAGS.out_dir, mode)))


if __name__ == '__main__':
  app.run(main)
