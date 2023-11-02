from typing import Dict, Type

from .abstract import AbsGame


class GameRegistry:

    def __init__(self):
        self.registry: Dict[str, Type[AbsGame]] = {}

    def add(self, game: Type[AbsGame]):
        self.registry[game.name] = game


game_registry = GameRegistry()
