import warnings

warnings.warn(
    "verticapy.learn.tsa.tools has been deprecated. It will be removed in v0.5.1. Use verticapy.stats instead",
    Warning,
)
from verticapy.stats import *
