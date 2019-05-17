"""Fake GCS client."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Generator, Text

import os
import shutil


class FakeBlob():  # pylint: disable=too-few-public-methods
  """Fake gcs.Blob"""

  def __init__(self, id: Text):  # pylint: disable=redefined-builtin
    self.id = id  # pylint: disable=invalid-name,redefined-builtin

  def download_to_filename(self, filename: Text):
    """Copy the blob to a file."""
    shutil.copyfile(self.id, filename)


class FakeBucket():  # pylint: disable=too-few-public-methods
  """Fake GCS bucket."""

  def __init__(self, dirname: Text):
    self.dirname = dirname

  def blob(self, bid: Text) -> FakeBlob:
    """Fetch a blob from the name."""
    return FakeBlob(bid)

  def list_blobs(self, prefix: Text) -> Generator[FakeBlob, None, None]:
    """List the blobs in the bucket."""
    for (dirpath, _, filenames) in os.walk(self.dirname):
      dirpath = dirpath[len(self.dirname) + 1:]
      for filename in filenames:
        filename = os.path.join(dirpath, filename)
        if filename.startswith(prefix):
          fullname = os.path.join(self.dirname, filename)
          yield FakeBlob(fullname)


class FakeClient():  # pylint: disable=too-few-public-methods
  """Fake GCS client."""

  def __init__(self, tmp_dir: Text):
    self.tmp_dir = tmp_dir

  def get_bucket(self, name: Text) -> FakeBucket:
    """Get the bucket by name."""
    return FakeBucket(os.path.join(self.tmp_dir, name))
