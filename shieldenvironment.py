import abc
from tf_agents.environments import py_environment
from tf_agents.trajectories.time_step import TimeStep

class ShieldedEnvironment(py_environment.PyEnvironment):
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def _step(self, action, save = False) -> TimeStep:
        pass

    @abc.abstractmethod
    def revert(self):
      pass

    @abc.abstractmethod
    def getAbsoluteScore(self) -> TimeStep:
        pass
