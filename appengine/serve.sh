#!/bin/sh

gunicorn --timeout=300 main:app
