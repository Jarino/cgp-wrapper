# Python wrapper for Cartesian Genetic Programming

**Note: still in development.**

Python wrapper for [CGP-Library](https://github.com/AndrewJamesTurner/CGP-Library) written in C. Wrapper is written in Cython and provides a scikit-learn like interface. Requires the [CGP-Library](https://github.com/AndrewJamesTurner/CGP-Library) to be installed in the system. 

## Install

Download and run:

```
make cython
```

Or directly:

```
python setup.py build_ext --inplace
```

## Test

Depends on [pytest](https://docs.pytest.org/en/latest/) library. To run test suite:
```
make test
```

Or directly:
```
pytest
```
