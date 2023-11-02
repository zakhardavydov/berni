from typing import Optional

from ..structures.action import Action
from .base import BaseAgent
from .registry import agent_registry

######
# 
#  __          __                 _        _____             _____                                                   
#  \ \        / /                | |      |_   _|           |  __ \                                                  
#   \ \  /\  / /    ___    _ __  | | __     | |    _ __     | |__) |  _ __    ___     __ _   _ __    ___   ___   ___ 
#    \ \/  \/ /    / _ \  | '__| | |/ /     | |   | '_ \    |  ___/  | '__|  / _ \   / _` | | '__|  / _ \ / __| / __|
#     \  /\  /    | (_) | | |    |   <     _| |_  | | | |   | |      | |    | (_) | | (_| | | |    |  __/ \__ \ \__ \
#      \/  \/      \___/  |_|    |_|\_\   |_____| |_| |_|   |_|      |_|     \___/   \__, | |_|     \___| |___/ |___/
#                                                                                     __/ |                          
#                                                                                    |___/                           
######
class QLAgentTrackOppon(BaseAgent):
    """
    The agent function takes in one argument:
    strategy - a function that outputs what structures an agent should take
    env - the environment the game is being played in
    id - the id number of the agent (more for debugging purposes)

    Initialises:
    memory - stores all the agent interactions in order with other agent (in this case it is for one round)
    score - the total score of the agent gathered in this iteration
    opponent - stores the current opponent
    structures - stores the current structures
    last - stores the last structures
    """

    name = "qlearning_track_opponent"

    def __init__(
            self,
            env,
            id: int,
            strategy=None,
            lr: float = 0.1,
            discount: float = 0.9,
            exploration_proba: float = 0.2,
            norm_weight: float = 1
    ):
        super().__init__(env=env, id=id, strategy=strategy, norm_weight=norm_weight)

        # added stuff, get a state as a tuple of the prev move and the reward
        self.internal_state: Optional[tuple] = None
        self.qtable = {}
        self.lr = lr
        self.discount = discount
        self.exploration_proba = exploration_proba

    def observe(self, reward, state):
        """
        Takes in the state space
        as of right now only the reward is implemented
        The observe function could be used to get the structures of
        """
        super().observe(reward, state)
        self.update()

    def update(self):

        history = self.get_last_history(buffer_size=1)
        current = history[0]

        new_state = (current.action, current.reward)
        prev_value = self.qtable.get((self.internal_state, current.action, current.opponent), 0)
        # qmax is the estimate of optimal future value, highest value on the table considering current state (the move it just did)
        qmax = max(self.qtable.get((new_state, action, current.opponent),0) for action in Action)
        # change the value in the qtable for the move it just did
        self.qtable[(self.internal_state, self.action, current.opponent)] = (1 - self.lr) * prev_value + self.lr * \
            (current.reward + self.discount * qmax - prev_value)
        # set the state to the new state
        self.internal_state = new_state
        
        

# agent_registry.add(QLAgentTrackOppon)
