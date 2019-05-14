# Build notes.
#
# pip install "apache-beam[gcp]" --ignore-installed PyYAML
#
# MacOS install of pyresample may fails with obscure errors, like
# #include <complex> fails, and the option -fopenmp is not supported.
# I fixed this by installing llvm.
#
# $ brew install llvm gcc
# $ export CC=/usr/local/opt/gcc/bin/gcc-9 CXX=/usr/local/opt/gcc/bin/g++-9
# $ pip install pyresample

.PHONY: typecheck lint run_local run_dataflow test

typecheck:
	pytype -P . `find . -name "*.py" -print`

lint:
	pylint `find . -name "*.py" -print`

run_local:
	python -m goes_truecolor.beam.make_truecolor_examples \
		--train_start_date="1/1/2018 17:00" \
		--train_end_date="1/1/2018 17:00" \
		--test_start_date="2/1/2018 17:00" \
		--test_end_date="1/1/2018 17:00" \
		--num_shards=1

run_dataflow:
	python -m goes_truecolor.beam.make_truecolor_examples --runner=DataflowRunner

test:
	python -m goes_truecolor.tests.goes_reader_test
	python -m goes_truecolor.tests.make_truecolor_examples_test
