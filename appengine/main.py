import datetime
import json
import logging
import os
import jinja2
import urllib
import webapp2

from google.appengine.ext import ndb

import site_manager

JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(
        os.path.join(os.path.dirname(__file__), 'contents')),
    extensions=['jinja2.ext.autoescape', 'jinja2.ext.do'],
    autoescape=True)

class HtmlPage(webapp2.RequestHandler):
  def get(self):
    name = urllib.unquote(self.request.path)
    if name == '/':
      name = '/index.html';

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
    }

    template = JINJA_ENV.get_template(name)
    self.response.write(template.render(template_values))

class MaskPage(webapp2.RequestHandler):
  def get(self):
    name = urllib.unquote(self.request.path)

    # Create the site manager.
    manager = site_manager.SiteManager(name)
    self.response.content_type = 'image/jpeg'
    self.response.write(manager.cloud_mask())

ROUTES = [
    ('[/]?', HtmlPage),
    ('/cloud_masks/.*\.jpg', MaskPage),
    ('.*\.html', HtmlPage),
]

application = webapp2.WSGIApplication(ROUTES, debug=True)
