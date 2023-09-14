from evofr.models import PianthamModel
from evofr import discretise_gamma


def test_init():
    gen = discretise_gamma(mn=4.2, std=2.0)
    model = PianthamModel(gen=gen)
