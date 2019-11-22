# Build notes.
#
# pip install "apache-beam[gcp]" --ignore-installed PyYAML
#
# MacOS install of pyresample may fails with obscure errors, like
# #include <complex> fails, and the option -fopenmp is not supported.
# I fixed this by installing llvm.
#
# $ brew install llvm gcc
# $ export CC=`which gcc-9` CXX=`which g++-9`
# $ pip install pyresample

.PHONY: typecheck lint run_local run_dataflow test

typecheck:
	pytype -P . goes_truecolor/beam/make_truecolor_examples.py
	pytype -P . goes_truecolor/beam/make_cloud_mask.py

lint:
	pylint `find goes_truecolor -name "*.py"`

run_truecolor_local:
	python -m goes_truecolor.beam.make_truecolor_examples \
		--train_start_date="1/1/2018 17:00" \
		--train_end_date="1/1/2018 17:00" \
		--test_start_date="2/1/2018 17:00" \
		--test_end_date="2/1/2018 17:00" \
		--num_shards=1

run_truecolor_dataflow:
	python -m goes_truecolor.beam.make_truecolor_examples --runner=DataflowRunner

run_cloud_masks_local:
	python -m goes_truecolor.beam.make_cloud_masks \
		--start_date="5/26/2019" \
		--end_date="11/20/2019"

run_cloud_masks_dataflow:
	python -m goes_truecolor.beam.make_cloud_masks \
		--start_date="5/20/2019" \
		--end_date="6/10/2019" \
		--max_workers=100 \
		--runner=DataflowRunner

test:
	python -m goes_truecolor.lib.goes_predict_test
	python -m goes_truecolor.lib.goes_reader_test
	python -m goes_truecolor.beam.make_truecolor_examples_test
	python -m goes_truecolor.beam.make_cloud_masks_test
