from evofr import discretise_gamma
from evofr.models import PianthamModel


def test_init():
    gen = discretise_gamma(mn=4.2, std=2.0)
    model = PianthamModel(gen=gen)
