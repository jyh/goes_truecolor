import datetime
import flask
import logging
import re
import urllib.parse

from typing import Text

import site_manager


DATE_REGEX = r'(\d+)-(\d+)-(\d+)'

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
@app.route('/<int:year>/<int:month>/<int:day>/', methods=['GET'])
def index(year: int = 2019, month: int = 1, day: int = 1) -> Text:
  t = datetime.datetime(year, month, day)
  name = t.strftime('/%Y/%m/%d/index.html')
  manager = site_manager.SiteManager(name)
  return flask.render_template('index.html', date=t, site_manager=manager)


@app.route('/', methods=['POST'])
@app.route('/<int:year>/<int:month>/<int:day>/', methods=['POST'])
def change_date(year: int = 2019, month: int = 1, day: int = 1) -> Text:
  date = flask.request.form['date']
  m = re.match(DATE_REGEX, date)
  t = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
  return flask.redirect(flask.url_for('index', year=t.year, month=t.month, day=t.day))


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
