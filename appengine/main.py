import datetime
import json
import logging
import os
import jinja2
import urllib
import re
import webapp2

from google.appengine.ext import ndb

import site_manager

JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(
        os.path.join(os.path.dirname(__file__), 'contents')),
    extensions=['jinja2.ext.autoescape', 'jinja2.ext.do'],
    autoescape=True)

DATE_REGEX = r'(\d+)-(\d+)-(\d+)'
PAGE_REGEX = r'^/(\d+)/(\d+)/(\d+)/'


class HtmlPage(webapp2.RequestHandler):
  def post(self):
    # Current article.
    name = urllib.unquote(self.request.path)

    # Comment text.
    date = self.request.get('date')
    m = re.match(DATE_REGEX, date)
    t = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # Redirect to the index page.
    self.redirect(t.strftime('/%Y/%m/%d/index.html'))

  def get(self):
    name = urllib.unquote(self.request.path)
    if name == '/':
      name = '/2019/1/1/index.html';

    # Date for the current page.
    m = re.match(PAGE_REGEX, name)
    t = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else datetime.datetime(2019, 1, 1)

    # Create the site manager.
    manager = site_manager.SiteManager(name)

    # Create context.
    url_prefix = 'http://' + self.request.environ['SERVER_NAME']
    port = self.request.environ['SERVER_PORT']
    if port:
      url_prefix += ':%s' % port

    template_values = {
        'site_url': url_prefix,
        'site_manager': manager,
        'get': self.request.GET,
        'date': t,
    }

    template = JINJA_ENV.get_template('/index.html')
    self.response.write(template.render(template_values))

class JpegPage(webapp2.RequestHandler):
  def get(self):
    name = urllib.unquote(self.request.path)

    # Create the site manager.
    manager = site_manager.SiteManager(name)
    img = manager.cloud_mask_jpeg()

    self.response.content_type = 'image/jpeg'
    self.response.write(img)

class GifPage(webapp2.RequestHandler):
  def get(self):
    name = urllib.unquote(self.request.path)

    # Create the site manager.
    manager = site_manager.SiteManager(name)
    img = manager.animated_gif()

    self.response.content_type = 'image/gif'
    self.response.write(img)

ROUTES = [
    ('[/]?', HtmlPage),
    ('/cloud_masks/.*\.jpg', JpegPage),
    ('/cloud_masks/.*\.gif', GifPage),
    ('.*\.html', HtmlPage),
]

application = webapp2.WSGIApplication(ROUTES, debug=True)
