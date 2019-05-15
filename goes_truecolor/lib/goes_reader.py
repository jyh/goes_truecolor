"""Reader for GOES satellite data on Google Cloud Storage."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Optional, Text, Tuple

import logging
import os
import re
import tempfile
import urllib

import collections
import datetime
import dateutil.tz
import numpy as np
import pyresample
import skimage
import skimage.io
import skimage.transform
import xarray

import google.cloud.storage as gcs

from goes_truecolor.lib import file_util

GOES_BUCKET = 'gcp-public-data-goes-16'

# Metadata fields to save from each imager file.
METADATA_KEYS = [
    'x',
    'y',
    'kappa0',
    'band_id',
    'x_image_bounds',
    'y_image_bounds',
    'goes_imager_projection',
    'time_coverage_start',
]

UTC = dateutil.tz.tzutc()

MAX_COLOR_VALUE = 255.9

IR_SCALE_FACTOR = [
    1e-2,  # 0
    1e-2,  # 1
    1e-2,  # 2
    1e-2,  # 3
    1e-2,  # 4
    1e-2,  # 5
    1e-2,  # 6
    1e-2,  # 7
    1 / 10, # 8
    1 / 20,  # 9
    1 / 30,  # 10
    1 / 100,  # 11
    1 / 100,  # 12
    1 / 150,  # 13
    1 / 150,  # 14
    1 / 150,  # 15
    1 / 150,  # 16
]

IR_CHANNELS = list(range(8, 17))

FILE_REGEX = (r'.*/OR_([^/]+)-M(\d+)C(\d\d)_G\d\d_s(\d+)(\d)'
              r'_e(\d+)(\d)_c(\d+)(\d)[.]nc')

GoesFile = collections.namedtuple('GoesFile', [
    'path', 'product', 'mode', 'channel',
    'start_date', 'end_date', 'creation_date'
])


def _parse_filename(filename: Text) -> GoesFile:
  """Convert a filename to a GoesFile description.

  Args:
    filename: string name of the file

  Returns:
    A GoesFile tuple corresponding to the filename

  Raises:
    ValueError: if the filename does not have the expected format.
  """
  m = re.match(FILE_REGEX, filename)
  if not m:
    raise ValueError(
        'Goes filename does not match regular expression: ' + filename)
  product = m.group(1)
  mode = int(m.group(2))
  channel = int(m.group(3))
  start_date = datetime.datetime.strptime(m.group(4), '%Y%j%H%M%S')
  start_date += datetime.timedelta(milliseconds=100 * int(m.group(5)))
  start_date = start_date.replace(tzinfo=UTC)
  end_date = datetime.datetime.strptime(m.group(6), '%Y%j%H%M%S')
  end_date += datetime.timedelta(milliseconds=100 * int(m.group(7)))
  end_date = end_date.replace(tzinfo=UTC)
  creation_date = datetime.datetime.strptime(m.group(8), '%Y%j%H%M%S')
  creation_date += datetime.timedelta(milliseconds=100 * int(m.group(9)))
  creation_date = creation_date.replace(tzinfo=UTC)
  return GoesFile(filename, product, mode, channel, start_date, end_date,
                  creation_date)


def resample_world_img(
    world_img: np.ndarray, new_grid: pyresample.geometry.AreaDefinition) -> np.ndarray:
  """Resample an image of the world to fit a pyresample area.

  Args:
    world_img: a flat image with lat/lng extent (-180, 180, 90, -90).
    new_grid: the desired resampling grid.

  Returns:
    A resampled image.
  """
  h, w = world_img.shape[:2]
  lats = np.linspace(90, -90, h)
  lons = np.linspace(-180, 180, w)
  mesh_lons, mesh_lats = np.meshgrid(lons, lats)
  base_grid = pyresample.geometry.GridDefinition(
      lons=mesh_lons, lats=mesh_lats)
  return pyresample.kd_tree.resample_nearest(
      base_grid, world_img, new_grid,
      radius_of_influence=50000)


def goes_area_definition(
    md: Dict[Text, Any], shape: Optional[Tuple[int, int]]) -> pyresample.geometry.AreaDefinition:
  """Get the area definition for the satellite image.

  Args:
    md: the satellite metadata
    shape: optional shape override

  Returns:
    A pyresample AreaDefinition for the satellite projection.
  """
  # Dee the following references for GOES imager.
  #  Ref-1: https://proj4.org/usage/projections.html
  #  Ref-2: https://proj4.org/operations/projections/geos.html
  proj_lon_0 = md['goes_imager_projection'].longitude_of_projection_origin
  proj_h_0_m = md['goes_imager_projection'].perspective_point_height  # meters
  if shape is None:
    nx = md['x'].shape[0]
    ny = md['y'].shape[0]
  else:
    ny, nx = shape
  x1, x2 = md['x_image_bounds'].data * proj_h_0_m
  y2, y1 = md['y_image_bounds'].data * proj_h_0_m
  extents_m = [x1, y1, x2, y2]
  grid = pyresample.geometry.AreaDefinition(
      'geos',
      'goes_conus',
      'geos',
      {'proj': 'geos',  # 'geostationary'
       'units': 'm',  # 'meters'
       'h': str(proj_h_0_m),  # height of the view point above Earth
       'lon_0': str(proj_lon_0),  # longitude of the proj center
       'a': str(md['goes_imager_projection'].semi_major_axis),
       'b': str(md['goes_imager_projection'].semi_minor_axis),
       'sweep': str(md['goes_imager_projection'].sweep_angle_axis),
      },
      nx,
      ny,
      extents_m)
  return grid


class GoesReader(object):  # pylint: disable=useless-object-inheritance
  """Client for accessing GOES-16 data."""

  # pylint: disable=too-many-instance-attributes
  def __init__(
      self,
      project_id: Text, goes_bucket_name: Text = GOES_BUCKET,
      key: Text = 'Rad', shape: Tuple[int, int] = (512, 512),
      tmp_dir: Optional[Text] = None,
      client: Optional[gcs.Client] = None):
    """Create a GoesReader.

    Args:
      project_id: the GCS project ID (for billing)
      goes_bucket_name: the GCS GOES bucket name (defaults to GOES-16)
      key: the data field (default 'Rad')
      shape: desired image shape
    """
    self.project_id = project_id
    self.goes_bucket_name = goes_bucket_name
    self.client = gcs.Client(project=project_id) if client is None else client
    self.tmp_dir = tempfile.mkdtemp('GoesReader') if tmp_dir is None else tmp_dir
    self.key = key
    self.shape = shape
    self.world_imgs = {}
    self.cache = {}

  def list_time_range(
      self, start_time: datetime.datetime, end_time: datetime.datetime) -> List[
          Tuple[datetime.datetime, Dict[int, gcs.Blob]]]:
    """List the blobs for GOES images within the given time range.

    Args:
      start_time: the beginning of the time range.
      end_time: the end of the time range.

    Returns:
      A list of time,channels pairs, where channels is a dictionary
      mapping channing number to GCS blob.
    """
    start_time = start_time.astimezone(UTC)
    end_time = end_time.astimezone(UTC)

    bucket = self.client.get_bucket(self.goes_bucket_name)
    blobs = []
    t = start_time
    h = datetime.timedelta(hours=1)
    while t < end_time + h:
      prefix = t.strftime('ABI-L1b-RadF/%Y/%j/%H')
      blobs.extend(bucket.list_blobs(prefix=prefix))
      t += h

    # Index them.
    channels = {}
    for b in blobs:
      f = _parse_filename(b.id)
      if f.start_date < start_time or f.start_date >= end_time:
        continue
      channel_map = channels.setdefault(f.start_date, {})
      channel_map[f.channel] = b
    return sorted(channels.items())

  def _resample_image(self, nc: xarray.DataArray) -> np.ndarray:
    """Extract an image from the GOES data."""
    kappa0 = nc.kappa0.data
    if np.isnan(kappa0):
      kappa0 = IR_SCALE_FACTOR[nc.band_id.data[0]]
    img = nc[self.key].data
    img = img.astype(np.float32) * kappa0
    img = np.nan_to_num(img)
    img = np.minimum(1, np.maximum(0, img))
    img = skimage.transform.resize(img, self.shape, mode='reflect', anti_aliasing=True)
    return (img * MAX_COLOR_VALUE).astype(np.uint8)

  def _load_image(self, blob: gcs.Blob) -> Tuple[np.ndarray, Dict[Text, Any]]:
    bid = blob.id
    if bid in self.cache:
      return self.cache[bid]
    with file_util.mktemp(dir=self.tmp_dir, suffix='.nc') as infile:
      logging.info('downloading %s', bid)
      blob.download_to_filename(infile)
      logging.info('downloaded %s', bid)
      with xarray.open_dataset(infile) as nc:
        img = self._resample_image(nc)
        logging.info('resampled %s', bid)
        md = {}
        for k in METADATA_KEYS:
          if k in nc.data_vars or k in nc.coords:
            md[k] = nc[k].copy()

        v = (img, md)
        self.cache[bid] = v
        return v

  def load_channel_images(
      self, t: datetime.datetime, channels: List[int]) -> Optional[Dict[
          int, Tuple[np.ndarray, Dict[Text, Any]]]]:
    """Load the GOES channels.

    Args:
      t: the observation time.
      channels: the channels to load.

    Returns:
      A dictionary mapping channel number to pairs (img, md), where img is the
      channel image, and md is the metadata.
    """
    blobs = self.list_time_range(t, t + datetime.timedelta(hours=1))
    if not blobs:
      return None
    t, channel_table = blobs[0]
    imgs = {}
    for c in channels:
      if c in channel_table:
        img, md = self._load_image(channel_table[c])
      else:
        img = np.zeros(self.shape, dtype=np.uint8)
        md = {}
      imgs[c] = (img, md)
    return imgs

  def load_world_img_from_url(
      self,
      world_map: Text,
      grid: pyresample.geometry.AreaDefinition) -> np.ndarray:
    """Fetch the world map image from a URL.

    Args:
      world_map_url: URL for the world map image.
      grid: the pyresample AreaDefinition for resampling.

    Returns:
      A numpy RGB image.
    """
    img = self.world_imgs.get(world_map)
    if img is None:
      with file_util.mktemp(dir=self.tmp_dir, suffix='.jpg') as infile:
        urllib.request.urlretrieve(world_map, infile)
        img = skimage.io.imread(infile)
        img = resample_world_img(img, grid)
        self.world_imgs[world_map] = img
    return img

  def cloud_mask(self, t: datetime.datetime) -> Optional[np.ndarray]:
    """Construct a cloud mask image for the specified time.

    Args:
      world_map: the URL for the world map image.
      t: the datetime.

    Returns:
      An image containing the cloud mask.
    """
    imgs = self.load_channel_images(t, [1])
    if imgs is None:
      return None
    img, _ = imgs[1]
    img = np.sqrt(img.astype(np.float32) / 256)
    mask = 1 / (1 + np.exp(-10 * (img - 0.3)))
    return (img * mask * MAX_COLOR_VALUE).astype(np.uint8)

  def raw_image(self, t: datetime.datetime, channels: List[int]) -> Optional[Tuple[np.ndarray, Dict[Text, Any]]]:
    """Load the GOES channels and flatten them into a multi-channel image.

    Args:
      world_map: the URL for the world map image.
      t: the observation time.
      channels: the channels to load.

    Returns:
      A numpy array containing the channel data.
    """
    table = self.load_channel_images(t, channels)
    if table is None:
      return None
    imgs = []
    for c in channels:
      img, md = table[c]
      imgs.append(img)
    return np.stack(imgs, axis=-1), md

  def goes_area_definition(self, md: Dict[Text, Any]) -> pyresample.geometry.AreaDefinition:
    return goes_area_definition(md, self.shape)
