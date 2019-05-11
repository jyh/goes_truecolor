"""Generate Tensorflow training examples from satellite images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

import datetime
import dateutil
import logging
import os
from absl import app
from absl import flags

import apache_beam as beam
from apache_beam.metrics import Metrics
import dateparser
import netCDF4 as nc4
import numpy as np
import tensorflow as tf

import goeslib.goes_reader  # pylint: disable=unused-import
import learning.hparams


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
    'start_date', '1/1/2019',
    'Start date for collecting images')

flags.DEFINE_string(
    'end_date', '1/1/2019',
    'End date for collecting images (inclusive)')

flags.DEFINE_integer(
    'tile_size', 64, 'size of tensorflow example tiles')


FLAGS = flags.FLAGS

IR_CHANNELS = list(range(7, 17))


# pylint: disable=too-many-instance-attributes,abstract-method
class CreateTFExamples(beam.DoFn):
  """Client for creating TFExamples from GOES-16 images."""

  def __init__(
      self, project_id, goes_bucket_name,
      image_size, tile_size, world_map):
    # type: (Text, Text, Text, int, int, Text)
    """Create an example generator.

    Args:
      project_id: the the GCS project ID (for billing)
      goes_bucket_name: the GOES bucket name
      image_size: desired image size
      tile_size: size of tiles for the ML model
      world_map: URL for the world map
    """
    super(CreateTFExamples, self).__init__()
    self.project_id = project_id
    self.goes_bucket_name = goes_bucket_name
    self.shape = image_size, image_size
    self.tile_size = tile_size
    self.world_map = world_map
    self.reader = None
    self.images_counter = Metrics.counter(self.__class__, 'images')

  def get_reader(self):
    # type: () -> goeslib.goes_reader.GoesReader:
    """Return a GoesReader for processing the input."""
    # pylint: disable=reimported,redefined-outer-name
    import goeslib.goes_reader

    if self.reader is None:
      self.reader = goeslib.goes_reader.GoesReader(
          self.project_id, self.goes_bucket_name, shape=self.shape)
    return self.reader

  # pylint: disable=arguments-differ
  def process(self, t):
    # Fetch the truecolor and IR images.
    reader = self.get_reader()
    logging.info('creating truecolor image for %s', t)
    rgb = reader.truecolor_image(self.world_map, t)
    logging.info('creating IR image for %s', t)
    ir = reader.load_channels(t, IR_CHANNELS)
    self.images_counter.inc()

    # Split into tiles and generate tensorflow examples.
    logging.info('creating tiles for %s', t)
    rgb_rows = np.split(rgb, self.tile_size, axis=0)
    ir_rows = np.split(ir, self.tile_size, axis=0)
    for rgb_row, ir_row in zip(rgb_rows, ir_rows):
      rgb_tiles = np.split(rgb_row, self.tile_size, axis=1)
      ir_tiles = np.split(ir_row, self.tile_size, axis=1)
      for rgb_tile, ir_tile in zip(rgb_tiles, ir_tiles):
        features = {
            learning.hparams.TRUECOLOR_CHANNELS_FEATURE_NAME: tf.train.Feature(
                int64_list=tf.train.Int64List(value=rgb_tile.ravel())),

            learning.hparams.IR_CHANNELS_FEATURE_NAME:  tf.train.Feature(
                int64_list=tf.train.Int64List(value=ir_tile.ravel())),
        }
        example = tf.train.Example(features=tf.train.Features(feature=features))
        yield bytes(example.SerializeToString())


def main(unused_argv):
  """Beam pipeline to create examples."""
  # Dates to sample.
  utc = dateutil.tz.tzutc()
  start_date = dateparser.parse(FLAGS.start_date)
  start_date = start_date.replace(tzinfo=utc)
  end_date = dateparser.parse(FLAGS.end_date)
  end_date = end_date.replace(tzinfo=utc)
  dates = []
  t = start_date
  while t <= end_date:
    dates.append(t)
    t += datetime.timedelta(days=1)

  # Create the beam pipeline.
  options = {'staging_location': os.path.join(FLAGS.out_dir, 'tmp', 'staging'),
             'temp_location': os.path.join(FLAGS.out_dir, 'tmp'),
             'job_name': datetime.datetime.now().strftime('truecolor-%y%m%d-%H%M%S'),
             'project': FLAGS.project,
             'max_num_workers': FLAGS.max_workers,
             'setup_file': './setup.py',
             'teardown_policy': 'TEARDOWN_ALWAYS',
             'no_save_main_session': True}
  opts = beam.pipeline.PipelineOptions(flags=[], **options)

  # Run the beam pipeline.
  with beam.Pipeline(FLAGS.runner, options=opts) as p:
    (p  # pylint: disable=expression-not-assigned
     | beam.Create(dates)
     | beam.ParDo(CreateTFExamples(
         FLAGS.project, FLAGS.goes_bucket,
         FLAGS.image_size, FLAGS.tile_size, FLAGS.world_map))
     | beam.io.tfrecordio.WriteToTFRecord(FLAGS.out_dir))

    job = p.run()
    if FLAGS.runner == 'DirectRunner':
      job.wait_until_finish()


if __name__ == '__main__':
  app.run(main)
