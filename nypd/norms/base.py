from .abstract import AbsNorm


class BaseNorm(AbsNorm):
    def __init__(self, reward: float = 2) -> None:
        super().__init__()
        self._reward = reward

    @property
    def reward(self) -> float:
        return self._reward
    
    @reward.setter
    def reward(self, value) -> None:
        self._reward = value

