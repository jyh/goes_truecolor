"""Reader for GOES satellite data on Google Cloud Storage."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import os
import tempfile



@contextmanager
def mktemp(**keyword_params):
  """Create a local file, removing it when the operation is complete.

  Args:
    **keyword_params: keyword params to be passed to tempfile.mkstemp().

  Yields:
    Filename of the temporary file.
  """

  fd, local_filename = tempfile.mkstemp(**keyword_params)
  os.close(fd)
  yield local_filename
  os.remove(local_filename)
