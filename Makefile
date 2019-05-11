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
	pytype -P . make_tfexamples.py

lint:
	pylint *.py */*.py

run_local:
	python make_tfexamples.py

run_dataflow:
	python make_tfexamples.py --runner=DataflowRunner
