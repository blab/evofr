from evofr.models import MultinomialLogisticRegression


def test_init():
    model = MultinomialLogisticRegression(tau=4.2)


def test_ols_feature():
    MultinomialLogisticRegression.make_ols_feature(start=0, stop=100)
