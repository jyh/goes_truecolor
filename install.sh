#!/bin/bash

set -x

export CC=`which gcc-9` CXX=`which g++-9`
pip install -U absl-py apache_beam[gcp] dateparser netcdf4 xarray pyresample google-api-core google-cloud-storage scikit-image tensorflow --ignore-installed PyYAML
