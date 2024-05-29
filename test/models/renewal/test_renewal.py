from evofr import (GARW, DirMultinomialSeq, RenewalModel, ZINegBinomCases,
                   discretise_gamma, discretise_lognorm, pad_delays)


def test_init():
    gen = discretise_gamma(mn=4.2, std=1.2)
    delays = pad_delays(
        [
            discretise_lognorm(mn=3.2, std=2.2),
            discretise_gamma(mn=3.2, std=2.0),
        ]
    )

    model = RenewalModel(
        gen,
        delays,
        14,
        0,
        RLik=GARW(0.1, 0.1),
        CLik=ZINegBinomCases(0.01),
        SLik=DirMultinomialSeq(100),
    )
