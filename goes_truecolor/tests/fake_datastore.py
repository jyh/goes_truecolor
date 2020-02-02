"""Fake Datastore client."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text


class FakeKey():  # pylint: disable=too-few-public-methods
  """Fake Datastore Key"""

  def __init__(self, ty: Text, val: Text):
    self.ty = ty  # pylint: disable=invalid-name
    self.val = val


class FakeEntity(dict):  # pylint: disable=too-few-public-methods
  """Fake datastore.Entity"""

  def __init__(self, key: FakeKey):
    super(FakeEntity, self).__init__()
    self.key = key


class FakeClient():  # pylint: disable=too-few-public-methods
  """Fake Datastore client."""

  def key(self, ty: Text, val: Text) -> FakeKey:  # pylint: disable=no-self-use
    """Get a Datastore key"""
    return FakeKey(ty, val)
