import datetime
import flask
import logging
import urllib.parse
import re

from typing import Text

import site_manager


app = flask.Flask(__name__)

@app.route('/')
@app.route('/<int:year>/<int:month>/<int:day>/')
def index(year: int = 2019, month: int = 1, day: int = 1) -> Text:
  t = datetime.datetime(year, month, day)
  name = t.strftime('/%Y/%m/%d/index.html')
  manager = site_manager.SiteManager(name)
  return flask.render_template('index.html', date=t, site_manager=manager)


@app.route('/cloud_masks/<int:year>/<int:day>/<string:filename>.jpg')
def handle_jpeg(year: int, day: int, filename: Text) -> flask.Response:
  name = urllib.parse.unquote(flask.request.path)
  manager = site_manager.SiteManager(name)
  img = manager.cloud_mask_jpeg()
  response = flask.make_response(img)
  response.headers.set('Content-Type', 'image/jpeg')
  return response  


@app.route('/cloud_masks/<int:year>/<int:month>/<int:day>/<string:filename>.gif')
def handle_gif(year: int, month: int, day: int, filename: Text) -> flask.Response:
  name = urllib.parse.unquote(flask.request.path)
  manager = site_manager.SiteManager(name)
  logging.info('Fetching %s', name)
  img = manager.animated_gif()
  response = flask.make_response(img)
  response.headers.set('Content-Type', 'image/gif')
  return response  


  
if __name__ == '__main__':
  # This is used when running locally only. When deploying to Google App
  # Engine, a webserver process such as Gunicorn will serve the app. This
  # can be configured by adding an `entrypoint` to app.yaml.
  app.run(host='localhost', port=8080, debug=True)
