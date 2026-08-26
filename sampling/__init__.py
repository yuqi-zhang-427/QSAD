# --*-- conding:utf-8 --*--
# @time:10/19/25 09:38
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:__init__.py.py

"""
sampling: quantum sampling utilities (simulator or IBM backends)

Public API:
- SamplingConfig, BackendConfig  (configuration)
- SamplingRunner                 (orchestrates sampling runs)
- make_backend                   (backend factory)
- build_ansatz, random_params, make_sampling_circuit  (circuit helpers)
"""

from .config import SamplingConfig, BackendConfig
from .sampler import SamplingRunner
from .backends import make_backend
from .circuits import build_ansatz, random_params, make_sampling_circuit

__all__ = [
    "SamplingConfig",
    "BackendConfig",
    "SamplingRunner",
    "make_backend",
    "build_ansatz",
    "random_params",
    "make_sampling_circuit",
]

__version__ = "0.1.0"
