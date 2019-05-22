import datetime
import io
import itertools
import logging
import os
import re

import numpy as np
from PIL import Image

import google.cloud.storage as gcs
from google.appengine.api import app_identity

ANIMATED_GIF_REGEX = r'.*/cloud_masks/(\d+)/(\d+)/(\d+)/animated\.gif'


class SiteManager(object):
  """Methods to manage the site."""

  def __init__(self, page_name):
    self.page_name = page_name
    self.bucket_name = 'weather-datasets'
    self.client = gcs.Client()

  def world_map(self):
    """Fetch the world map.

    For now, we assume the world map is static.
    """
    bucket = self.client.bucket(self.bucket_name)
    blob = bucket.blob('world_map.jpg')
    data = blob.download_as_string()
    f = io.BytesIO(data)
    img = Image.open(f)
    return np.array(img)

  def animated_gif(self):
    # Parse the path.
    m = re.match(ANIMATED_GIF_REGEX, self.page_name)
    if not m:
      raise ValueError(
        'filename does not match regular expression: {}'.format(self.page_name))
    t = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    tm = t.timetuple()

    # Get the world map.
    world_img = self.world_map()
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
    f = io.BytesIO()
    images[0].save(f, 'GIF', save_all=True, append_images=images[1:], duration=100, loop=0)
    return f.getvalue()
