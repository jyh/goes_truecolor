#!/bin/bash

export GCLOUD_PROJECT=weather-324

gunicorn main:app
