from sklearn import datasets
from sklearn.model_selection import cross_val_score

from pycgp import CGPClassifier



def test_fit_and_single_predict():
    """Test whether it fits without any error."""
    iris = datasets.load_iris()

    cls = CGPClassifier()

    cls.fit(iris.data, iris.target)


    result = cls.predict([1,2,3,4])
    assert True


def test_fit_and_multiple_predict():
    """Test of prediction for multiple input vectors."""
    iris = datasets.load_iris()

    cls = CGPClassifier()

    cls.fit(iris.data, iris.target)

    result = cls.predict([
        [1,2,3,4],
        [3,4,5,6]
    ])

    assert True

def test_cross_validation():
    iris = datasets.load_iris()

    cls = CGPClassifier()

    scores = cross_val_score(cls, iris.data, iris.target, cv=5, scoring='accuracy')

    print(scores)

    assert True, "cross validation ended up without runtime errors"
