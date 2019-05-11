# Build notes.
#
# pip install apache-beam"[gcp]" --ignore-installed PyYAML
#
# MacOS install of pyresample may fails with obscure errors, like
# #include <complex> fails, and the option -fopenmp is not supported.
# I fixed this by installing llvm.
#
# $ brew install llvm
# # export CC=/usr/local/opt/gcc/gcc-9 CXX=/usr/local/opt/gcc/g++-9
# $ pip install pyresample

.PHONY: typecheck lint run_local run_dataflow

typecheck:
	pytype -P . truecolor/preproc/make_truecolor_examples.py

lint:
	pylint truecolor/preproc/make_truecolor_examples.py

run_local:
	python -m truecolor.preproc.make_truecolor_examples

run_dataflow:
	python -m truecolor.preproc.make_truecolor_examples --runner=DataflowRunner
