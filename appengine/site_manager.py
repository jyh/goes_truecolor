import datetime
import io
import itertools
import logging
import os
import re
import tempfile

import numpy as np
from PIL import Image

import google.cloud.storage as gcs
from google.appengine.api import app_identity

ANIMATED_GIF_REGEX = r'.*/cloud_masks/(\d+)/(\d+)/(\d+)/animated\.gif'
ISO_TIME_REGEX = r'.*/(\d+)-(\d+)-(\d+).(\d+):(\d+):(\d+)([.](\d+))?([+]\d\d:\d\d)?[.]jpg'
FILE_TIME_REGEX = r'.*/(\d\d\d\d)(\d\d)(\d\d)_(\d\d)(\d\d)_(\d+)[.]jpg'


def _parse_date(s):
  m = re.match(ISO_TIME_REGEX, s)
  if m:
    return datetime.datetime(
        int(m.group(1)), int(m.group(2)), int(m.group(3)),
        int(m.group(4)), int(m.group(5)), int(m.group(6)),
        int(m.group(8)))

  m = re.match(FILE_TIME_REGEX, s)
  if m:
    return datetime.datetime(
        int(m.group(1)), int(m.group(2)), int(m.group(3)),
        int(m.group(4)), int(m.group(5)), 0,
        int(m.group(6)))

  raise ValueError('invalid date {}'.format(s))


class SiteManager(object):
  """Methods to manage the site."""

  def __init__(self, page_name):
    self.page_name = page_name
    self.bucket_name = 'weather-datasets'
    self.client = gcs.Client()

  def list_blobs(self, count=None):
    # Parse the path.
    m = re.match(ANIMATED_GIF_REGEX, self.page_name)
    if m:
      t = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
      t = datetime.datetime(2019, 1, 1)
    tm = t.timetuple()

    # Get all the files for the day.
    bucket = self.client.bucket(self.bucket_name)
    prefix = 'cloud_masks/{:04d}/{:03d}'.format(tm.tm_year, tm.tm_yday)
    names = ['/' + b.name for b in bucket.list_blobs(prefix=prefix)]

    # If there is a desired count, sample the names.
    if count is not None:
      n = len(names)
      names = [names[(i * n) // count] for i in range(count)]

    # Parse the times
    return [(i, _parse_date(name), name) for i, name in enumerate(names)]

  def world_img(self):
    """Fetch the world map.

    For now, we assume the world map is static.
    """
    bucket = self.client.bucket(self.bucket_name)
    blob = bucket.blob('world_map.jpg')
    data = blob.download_as_string()
    f = io.BytesIO(data)
    img = Image.open(f)
    return np.array(img)

  def cloud_mask_jpeg(self):
    # Get the world map.
    world_img = self.world_img()
    world_img = np.array(world_img, dtype=np.float32) / 256

    # Get all the files for the day.
    bucket = self.client.bucket(self.bucket_name)
    blob = bucket.blob(self.page_name[1:])
    s = blob.download_as_string()
    f = io.BytesIO(s)
    lum = Image.open(f)
    lum = np.array(lum)
    lum = lum.astype(np.float32) / 256
    lum = lum[:, :, np.newaxis]
    mask = 1 / (1 + np.exp(-10 * (lum - 0.3)))
    img = lum * mask + (1 - mask) * world_img
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)

    # The old PIL used by AppEngine can only save to a real file, not BytesIO.
    fd = tempfile.TemporaryFile(suffix='.jpg')
    img.save(fd, 'JPEG')
    fd.seek(0)
    s = fd.read()
    fd.close()
    return s

  def animated_gif(self):
    # Parse the path.
    m = re.match(ANIMATED_GIF_REGEX, self.page_name)
    if not m:
      raise ValueError(
        'filename does not match regular expression: {}'.format(self.page_name))
    t = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    tm = t.timetuple()

    # Get the world map.
    world_img = self.world_img()
    world_img = np.array(world_img, dtype=np.float32) / 256

    # Get all the files for the day.
    bucket = self.client.bucket(self.bucket_name)
    prefix = 'cloud_masks/{:04d}/{:03d}'.format(tm.tm_year, tm.tm_yday)
    blobs = bucket.list_blobs(prefix=prefix)

    # Image compositing.
    images = []
    for b in itertools.islice(blobs, 0, None, 4):
      s = b.download_as_string()
      f = io.BytesIO(s)
      lum = Image.open(f)
      lum = np.array(lum)
      lum = lum.astype(np.float32) / 256
      lum = lum[:, :, np.newaxis]
      mask = 1 / (1 + np.exp(-10 * (lum - 0.3)))
      img = lum * mask + (1 - mask) * world_img
      img = (img * 255).astype(np.uint8)
      img = Image.fromarray(img)
      images.append(img)
    if not images:
      raise ValueError('no image {}'.format(self.page_name))

    # Old PIL can only save to a real file, bot BytesIO, and appengine can only
    # use TemporaryFile.
    logging.error('Pillow version %s', Image.PILLOW_VERSION)
    fd = tempfile.TemporaryFile(suffix='.gif')
    images[0].save(fd, 'GIF', save_all=True, append_images=images[1:], duration=100, loop=0)
    fd.seek(0)
    s = fd.read()
    fd.close()
    return s
