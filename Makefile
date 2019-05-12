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

.PHONY: typecheck lint run_local run_dataflow

typecheck:
	pytype -P . `find . -name "*.py" -print`

lint:
	pylint `find . -name "*.py" -print`

run_local:
	python -m goes_truecolor.preproc.make_truecolor_examples

run_dataflow:
	python -m goes_truecolor.preproc.make_truecolor_examples --runner=DataflowRunner

test:
	python -m goes_truecolor.tests.goes_reader_test
	python -m goes_truecolor.tests.make_truecolor_examples_test
