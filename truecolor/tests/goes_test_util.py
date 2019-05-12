"""Utilities for creating fake GCS files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

import os

import datetime
import h5py
import numpy as np


def goes_filename(t: datetime.datetime, channel: int) -> Text:
  """Create a GOES filename from time and channel."""
  create_date = t.strftime('%Y%j%H%M%S0')
  dirname = t.strftime('ABI-L1b-RadF/%Y/%j/%H')
  filename = ('OR_ABI-L1b-RadF-M3C{channel:02d}_G16_s'
              '{create_date}_e{create_date}_c{create_date}.nc'.format(
                  channel=channel, create_date=create_date))
  return os.path.join(dirname, filename)


def create_fake_goes_image(tmp_dir: Text, t: datetime.datetime, channel: int) -> Text:
  """Create a fake GOES image in a tmp directory.

  Args:
    tmp_dir: the temp directory.
    t: the date of the image.
    channel: the channel

  Returns:
    The filename.
  """
  # Create the directory.
  filename = os.path.join(tmp_dir, goes_filename(t, channel))
  dirname = os.path.dirname(filename)
  if not os.path.exists(dirname):
    os.makedirs(dirname)

  # Create the content.
  w = 128
  x = np.arange(0, w)
  y = np.arange(0, w)
  xv, yv = np.meshgrid(x, y)
  data = xv + yv

  # Write it to the file.
  with h5py.File(filename, 'w') as f:
    f.create_dataset('x', data=x)
    f.create_dataset('y', data=y)
    f.create_dataset('x_image_bounds', data=[-0.15187199, 0.15187199])
    f.create_dataset('y_image_bounds', data=[0.15187199, -0.15187199])
    f.create_dataset('Rad', data=data)
    f.create_dataset('band_id', data=channel)
    f.create_dataset('kappa0', data=[1e-2])
    f.attrs['time_coverage_start'] = t.strftime('%Y/%m/%d %H:%MUTC')
    proj = f.create_dataset('goes_imager_projection', [0])
    proj.attrs['semi_major_axis'] = 6378137.0
    proj.attrs['semi_minor_axis'] = 6356752.31414
    proj.attrs['sweep_angle_axis'] = 'x'
    proj.attrs['longitude_of_projection_origin'] = -75
    proj.attrs['perspective_point_height'] = 35786000

  return filename
