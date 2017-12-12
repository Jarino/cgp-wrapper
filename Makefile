cython:
	python setup.py build_ext --inplace

test:
	pytest
