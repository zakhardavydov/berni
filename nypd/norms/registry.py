from typing import Dict, Type

from .abstract import AbsNorm


class NormRegistry:

    def __init__(self):
        self.registry: Dict[str, Type[AbsNorm]] = {}

    def add(self, norm: Type[AbsNorm]):
        self.registry[norm.name] = norm


norm_registry = NormRegistry()
