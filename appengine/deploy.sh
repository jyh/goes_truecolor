#!/bin/sh

export CLOUDSDK_PYTHON=python2.7
gcloud app deploy --project=weather-324 app.yaml
