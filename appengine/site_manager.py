import datetime
import io
import itertools
import logging
import os
import re
import tempfile

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from typing import Optional, List, Text, Tuple

import google.cloud.storage as gcs

PAGE_REGEX = r'.*/(\d+)/(\d+)/(\d+)/[^/]+[.]html'
ANIMATED_GIF_REGEX = r'.*/cloud_masks/(\d+)/(\d+)/(\d+)/animated\.gif'
ISO_TIME_REGEX = r'.*/(\d+)-(\d+)-(\d+).(\d+):(\d+):(\d+)([.](\d+))?([+]\d\d:\d\d)?[.]jpg'
FILE_TIME_REGEX = r'.*/(\d\d\d\d)(\d\d)(\d\d)_(\d\d)(\d\d)_(\d+)[.]jpg'

# AppEngine has pretty severe memory restrictions, so limit the number of frames to
# add to an animated gif.
ANIMATED_GIF_MAX_FRAMES = 100


def _parse_date(s: Text) -> datetime.datetime:
  """Convert a filename to a datetime."""
  m = re.match(ISO_TIME_REGEX, s)
  if m:
    ms = m.group(8)
    return datetime.datetime(
        int(m.group(1)), int(m.group(2)), int(m.group(3)),
        int(m.group(4)), int(m.group(5)), int(m.group(6)),
        int(ms if ms else 0))

  m = re.match(FILE_TIME_REGEX, s)
  if m:
    return datetime.datetime(
        int(m.group(1)), int(m.group(2)), int(m.group(3)),
        int(m.group(4)), int(m.group(5)), 0,
        int(m.group(6)))

  raise ValueError('invalid date {}'.format(s))


class SiteManager():
  """Methods to manage the site."""

  def __init__(self, page_name: Text):
    self.page_name = page_name
    self.bucket_name = 'weather-datasets'
    self.client = gcs.Client()

  def list_blobs(self, count: Optional[int] = None) -> List[Tuple[int, datetime.datetime, Text]]:
    """Return a list of cloud images for the current date.

    Args:
      count: the number of results to return.

    Returns:
      A sequence of triples (i, t, name), where <i> is the number of the item,
      <t> is its timestamp, and <name> is the name of he GCS blob.
    """
    # Parse the path.
    m = re.match(PAGE_REGEX, self.page_name)
    t = (datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m
         else datetime.datetime(2019, 1, 1))
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

  def world_img(self) -> np.ndarray:
    """Fetch the world map.

    For now, we assume the world map is static.
    """
    bucket = self.client.bucket(self.bucket_name)
    blob = bucket.blob('world_map.jpg')
    data = blob.download_as_string()
    f = io.BytesIO(data)
    img = Image.open(f)
    return np.array(img)

  def cloud_mask_jpeg(self) -> bytes:
    """Return the current image in JPEG format."""
    # Get the world map.
    world_img = self.world_img()
    world_img = np.array(world_img, dtype=np.float32) / 256

    # Get the mask file.
    bucket = self.client.bucket(self.bucket_name)
    blob = bucket.blob(self.page_name[1:])
    s = blob.download_as_string()
    f = io.BytesIO(s)
    lum = Image.open(f)
    lum = np.array(lum)
    lum = lum.astype(np.float32) / 256
    lum = lum[:, :, np.newaxis]

    # Generate the composite.
    mask = 1 / (1 + np.exp(-10 * (lum - 0.3)))
    img = lum * mask + (1 - mask) * world_img
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)

    # Convert to JPEG.
    f = io.BytesIO()
    img.save(f, 'JPEG')
    return f.getvalue()

  def animated_gif(self) -> bytes:
    """Return an animated GIF of images for the current day."""
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
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Limit the number of frames.
    n = len(blobs)
    if n > ANIMATED_GIF_MAX_FRAMES:
      blobs = [blobs[(i * n) // ANIMATED_GIF_MAX_FRAMES] for i in range(ANIMATED_GIF_MAX_FRAMES)]

    # Date labels
    font = ImageFont.truetype("arial.ttf", 50)

    # Image compositing.
    images = []
    for b in blobs:
      s = b.download_as_string()
      f = io.BytesIO(s)
      lum = Image.open(f)
      lum = np.array(lum)
      lum = lum.astype(np.float32) / 256
      lum = lum[:, :, np.newaxis]
      mask = 1 / (1 + np.exp(-10 * (lum - 0.3)))
      img = lum * mask + (1 - mask) * world_img
      img = (img * 255).astype(np.uint8)
      img = np.pad(img, ((50, 0), (0, 0), (0, 0)))
      img = Image.fromarray(img)

      # Add the date
      t = _parse_date(b.name)
      draw = ImageDraw.Draw(img)
      draw.text((0, 0), t.strftime('GOES 16 %Y/%m/%d %H:%M'), font=font)

      images.append(img)
    if not images:
      raise ValueError('no image {}'.format(self.page_name))

    # Produce animated GIF.  Note that appengine has severe memory limitations.
    f = io.BytesIO()
    images[0].save(f, 'GIF', save_all=True, append_images=images[1:], duration=100, loop=0)
    return f.getvalue()
