# evofr/registries.py
INFERENCE_REGISTRY = {}
PRIOR_REGISTRY = {}

def register_inference(cls):
    INFERENCE_REGISTRY[cls.__name__] = cls
    return cls

def register_prior(cls):
    PRIOR_REGISTRY[cls.__name__] = cls
    return cls
