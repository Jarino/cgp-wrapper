import pycgp

def test_run():
    """Test whether it run without any error."""
    pycgp.run([1,2,3,4,5], [1,2,3,4,5])
    assert True
