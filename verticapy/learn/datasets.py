import warnings

warnings.warn(
    "verticapy.learn.datasets has been deprecated. It will be removed in v0.5.1. Use verticapy.datasets instead",
    Warning,
)
from verticapy.datasets import *
