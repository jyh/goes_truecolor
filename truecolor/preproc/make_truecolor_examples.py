"""Generate Tensorflow training examples from satellite images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Generator, Text

import datetime
import logging
import os
from absl import app
from absl import flags

import apache_beam as beam
import dateparser
import dateutil
import multiprocessing
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
    'start_date', '1/1/2019',
    'Start date for collecting images')

flags.DEFINE_string(
    'end_date', '1/1/2019',
    'End date for collecting images (inclusive)')

flags.DEFINE_integer(
    'tile_size', 64, 'size of tensorflow example tiles')


FLAGS = flags.FLAGS

IR_CHANNELS = list(range(7, 17))

reader = None

# pylint: disable=too-many-instance-attributes,abstract-method
class CreateTFExamples(beam.DoFn):
  """Client for creating TFExamples from GOES-16 images."""

  def __init__(
      self, project_id, goes_bucket_name,
      image_size, tile_size, world_map):
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

  def get_reader(self):
    from truecolor.goeslib import goes_reader
    
    """Return a GoesReader for processing the input."""
    if reader is None:
      reader = goes_reader.GoesReader(
        self.project_id, self.goes_bucket_name, shape=self.shape)
    return reader

  # pylint: disable=arguments-differ,too-many-locals
  def process(self, t):
    import logging
    import numpy as np
    import tensorflow as tf

    # Fetch the truecolor and IR images.
    reader = self.get_reader()
    logging.info('creating truecolor image for %s', t)
    rgb = reader.truecolor_image(self.world_map, t)
    logging.info('creating IR image for %s', t)
    ir = reader.load_channels(t, IR_CHANNELS)

    # Split into tiles and generate tensorflow examples.
    logging.info('creating tiles for %s', t)
    rgb_rows = np.split(rgb, self.tile_size, axis=0)
    ir_rows = np.split(ir, self.tile_size, axis=0)
    for rgb_row, ir_row in zip(rgb_rows, ir_rows):
      rgb_tiles = np.split(rgb_row, self.tile_size, axis=1)
      ir_tiles = np.split(ir_row, self.tile_size, axis=1)
      for rgb_tile, ir_tile in zip(rgb_tiles, ir_tiles):
        features = {
            hparams.TRUECOLOR_CHANNELS_FEATURE_NAME: tf.train.Feature(
                int64_list=tf.train.Int64List(value=rgb_tile.ravel())),

            hparams.IR_CHANNELS_FEATURE_NAME:  tf.train.Feature(
                int64_list=tf.train.Int64List(value=ir_tile.ravel())),
        }
        example = tf.train.Example(features=tf.train.Features(feature=features))
        yield example.SerializeToString()


def make_truecolor_examples(t, project_id, goes_bucket_name, image_size, tile_size, world_map):
  import logging
  import numpy as np
  import tensorflow as tf
  from truecolor.goeslib import goes_reader
  
  # Fetch the truecolor and IR images.
  shape = image_size, image_size
  reader = goes_reader.GoesReader(project_id, goes_bucket_name, shape=shape)
  logging.info('creating truecolor image for %s', t)
  rgb = reader.truecolor_image(world_map, t)
  logging.info('creating IR image for %s', t)
  ir = reader.load_channels(t, IR_CHANNELS)
  
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
             'setup_file': os.path.join(
               os.path.dirname(os.path.abspath(__file__)), '../../setup.py'),
             'teardown_policy': 'TEARDOWN_ALWAYS',
             'save_main_session': True}
  opts = beam.pipeline.PipelineOptions(flags=[], **options)

  # Run the beam pipeline.
  with beam.Pipeline(FLAGS.runner, options=opts) as p:
    (p  # pylint: disable=expression-not-assigned
     | beam.Create(dates)
     | beam.FlatMap(make_truecolor_examples,
         FLAGS.project, FLAGS.goes_bucket,
         FLAGS.image_size, FLAGS.tile_size, FLAGS.world_map)
     | beam.io.tfrecordio.WriteToTFRecord(FLAGS.out_dir))


if __name__ == '__main__':
  app.run(main)
