import abc
from typing import Any, NamedTuple
import dm_env
from dm_env import StepType

from move import Move

class FeasibleEnvironment(dm_env.Environment):
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def obtainFeasibleMoves():
        pass