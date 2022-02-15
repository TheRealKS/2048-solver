from environment_spec import FeasibleEnvironment

class InfeasibilityChecker:

    def __init__(self, environment : FeasibleEnvironment) -> None:
        self.env = environment

    def getFeasibleMoves(self, observation):
        moves = self.env.obtainFeasibleMoves(observation)
        #Convert the moves to numeric values
        moves = map(lambda m: m.value, moves)
        return moves