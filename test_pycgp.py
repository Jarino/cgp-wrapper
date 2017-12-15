"""
Test suite for CGPWrapper
"""

from sklearn import datasets
from sklearn.model_selection import cross_val_score

from cgpwrapper import CGPClassifier


def test_fit_and_single_predict():
    """ Test whether it fits without any error."""
    iris = datasets.load_iris()

    cls = CGPClassifier()

    cls.fit(iris.data, iris.target)

    cls.predict([1, 2, 3, 4])
    assert True


def test_fit_and_multiple_predict():
    """ Test of prediction for multiple input vectors."""
    iris = datasets.load_iris()

    cls = CGPClassifier()

    cls.fit(iris.data, iris.target)

    cls.predict([
        [1, 2, 3, 4],
        [3, 4, 5, 6]
    ])

    assert True


def test_cross_validation():
    """ Test of successful sklearn cross validation run """
    iris = datasets.load_iris()

    cls = CGPClassifier()

    scores = cross_val_score(
        cls, iris.data, iris.target, cv=5, scoring='accuracy')

    print(scores)

    assert True, "cross validation ended up without runtime errors"


def test_variable_inputs():
    """ Test inputs of varying length """
    cls = CGPClassifier()

    data = [[1, 2], [2, 4], [4, 4]]
    target = [3, 4, 5]

    cls.fit(data, target)

    cls.predict([9, 9])

    data = [[1, 2, 3], [2, 4, 3], [4, 4, 3]]
    target = [3, 4, 5]

    cls.fit(data, target)

    cls.predict([9, 9])

    assert True


def test_variable_outputs():
    """ Test outputs of varying length """
    cls = CGPClassifier()

    data = [[1, 2], [2, 4], [4, 4]]
    target = [[1, 3], [0.4, 4], [3, 3]]

    cls.fit(data, target)

    result = cls.predict([4, 4])

    assert len(result) == 2

    data = [[1, 2, 3], [2, 4, 3], [4, 4, 3]]
    target = [[1, 5, 3], [9, 8, 0], [4, 5, 1]]

    cls.fit(data, target)

    result = cls.predict([9, 9, 9])

    assert len(result) == 3


def test_functions_as_params():
    """ Test whether functions assigned as parameters passes without error """
    cls = CGPClassifier(funset="add,mul,div,sub")

    data = [[1, 2], [2, 4], [4, 4]]
    target = [[1, 3], [0.4, 4], [3, 3]]

    cls.fit(data, target)

    cls.predict([5, 5])

    assert True
