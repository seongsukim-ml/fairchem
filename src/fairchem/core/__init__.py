# flake8: noqa: E402
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

# TODO: Remove this warning filter when torchtnt fixes pkg_resources deprecation warning.
warnings.filterwarnings(
    "ignore",
    message=(
        "pkg_resources is deprecated as an API. "
        "See https://setuptools.pypa.io/en/latest/pkg_resources.html."
    ),
    category=UserWarning,
)

from fairchem.core._config import clear_cache
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator


def _load_pretrained_module():
    try:
        from fairchem.core.calculate import pretrained_mlip as _pretrained_mlip
    except ModuleNotFoundError as err:  # pragma: no cover - defensive
        if err.name != "huggingface_hub":
            raise

        class _MissingPretrainedModule(ModuleType):
            def __getattr__(self, name):  # pragma: no cover - defensive
                raise ModuleNotFoundError(
                    "huggingface_hub is required for pretrained model utilities. "
                    "Install it with `pip install huggingface_hub` or install "
                    "fairchem with the appropriate extras."
                ) from err

        return _MissingPretrainedModule("fairchem.core.pretrained_mlip")

    return _pretrained_mlip


pretrained_mlip = _load_pretrained_module()

try:
    __version__ = version("fairchem.core")
except PackageNotFoundError:
    __version__ = ""

__all__ = [
    "FAIRChemCalculator",
    "pretrained_mlip",
    "clear_cache",
]
