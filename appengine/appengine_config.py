# appengine_config.py
from google.appengine.ext import vendor

# Add any libraries install in the "lib" folder.
vendor.add('lib')

try:
    from google.appengine.tools.devappserver2.python.runtime.stubs import FakeFile
    FakeFile._allowed_dirs.update(['/System/Library/CoreServices/'])
except ImportError:
    pass
