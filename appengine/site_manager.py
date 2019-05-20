import datetime
import logging
import os

import google.cloud.storage as gcs
from google.appengine.api import app_identity


class SiteManager(object):
  """Methods to manage the site."""

  def __init__(self, page_name):
    self.page_name = page_name
    self.bucket_name = 'weather-datasets'
    self.client = gcs.Client()

  def _list_date(self, year, doy):
    bucket = self.client.bucket(self.bucket_name)
    prefix = 'cloud_masks/{:04d}/{:03d}'.format(year, doy)
    blobs = bucket.list_blobs(prefix=prefix)
    return [b.name for b in blobs]

  def cloud_mask(self):
    bucket = self.client.bucket(self.bucket_name)
    blob = bucket.blob(self.page_name[1:])
    return blob.download_as_string()
