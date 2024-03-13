import os
import time
import json
import matplotlib
import matplotlib.pyplot as plt

from nypd.env import BaseEnv
from nypd.game import AbsGame
from nypd.norms import AbsNorm
from nypd.ps import AbsPartnerSelection
from nypd.structures import ExperimentSetup, Action
from nypd.agent import AbsAgent

from berni.assesment import AgentAssesor


matplotlib.use('Agg')


class LLMEnv(BaseEnv):

    def __init__(
            self,
            num_agents: int, 
            num_rounds: int,
            game: AbsGame,
            norm: AbsNorm,
            ps: AbsPartnerSelection,
            setup_assesment: list[AgentAssesor] | None = None,
            step_assesment: list[AgentAssesor] | None = None,
            complete_assesment: list[AgentAssesor] | None = None,
    ):
        self._setup_assesment = setup_assesment
        self._step_assessment = step_assesment
        self._complete_assessment = complete_assesment

        self.flips = {}

        self.bias = []
        self.bias_av = []

        self.results_dir = None

        super().__init__(num_agents, num_rounds, game, norm, ps)

    def setup(self):
        super().setup()
        if self._setup_assesment:
            for ass in self._setup_assesment:
                for agent in self.agents:
                    ass.assess(agent, agent._strategy)

    def evaluate_flip(self, agent1: AbsAgent, agent2: AbsAgent):
        history_item = agent1.get_last_history()
        if history_item:
            if history_item[0].action == Action.C:
                bias_gap = history_item[0].bias_gap
                if bias_gap in self.flips:
                    self.flips[bias_gap] += 1
                else:
                    self.flips[bias_gap] = 1

    def step(self):
        super().step()
        if self._step_assessment:
            for ass in self._step_assessment:
                for agent in self.agents:
                    ass.assess(agent)
        for i, j in self.pairs:
            agent1 = self.agents[i]
            agent2 = self.agents[j]
            self.evaluate_flip(agent1, agent2)
            self.evaluate_flip(agent2, agent1)
        total_bias = 0
        agent_bias_map = {}
        for agent in self.agents:
            total_bias += agent.bias_score
            agent_bias_map[agent.id] = {
                "bias_score": agent.bias_score,
                "initial_bias": agent.initial_bias,
                "round_prompt": agent.round_prompt,
                "round_neighbours": [
                    agent.opponent
                ],
                "round_neighbours_opinion": [
                    agent.opponent_model.opinion
                ],
                "round_neighbours_action": [
                    agent.action
                ],
                "round_action": agent.action,
                "outcome_opinion": agent.opinion
            }
        self.bias.append(total_bias)
        self.bias_av.append(total_bias / self.num_agents)
        save_path = f"{self.results_dir}/matrix"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = f"{save_path}/{self.rounds}.json"
        with open(save_path, "w") as f:
            json.dump(agent_bias_map, f)

    def _plot_flips(self, run):
        sorted_keys = sorted(self.flips.keys())
        sorted_values = [self.flips[key] for key in sorted_keys]

        # Plotting the histogram
        plt.bar(sorted_keys, sorted_values)
        plt.xlabel('Bias gap')
        plt.ylabel('Frequency')
        plt.title('Frequency of opinion flips based on bias gap')
        plt.xticks(sorted_keys)
        if not os.path.exists(f"nypd/artifacts/{run}"):
            os.makedirs(f"nypd/artifacts/{run}")
        plt.savefig(f"nypd/artifacts/{run}/flips.png", dpi=300, bbox_inches='tight')
        plt.clf()

    def _plot_bias(self, run):
        x = list(range(0, len(self.bias)))
        plt.plot(x, self.bias)
        plt.title("Total Bias Over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Totla Bias")
        plt.savefig(f"nypd/artifacts/{run}/bias.png", dpi=300, bbox_inches='tight')

    def complete(self, run=None):
        super().complete()
        self._plot_flips(run)
        self._plot_bias(run)
        if self._complete_assessment:
            for ass in self._complete_assessment:
                for agent in self.agents:
                    ass.assess(agent, agent._strategy)

    @staticmethod
    def from_exp_setup(
        exp_setup: ExperimentSetup,
        ps: AbsPartnerSelection,
        setup_assesment: list[AgentAssesor] | None = None,
        step_assesment: list[AgentAssesor] | None = None,
        complete_assesment: list[AgentAssesor] | None = None
    ):
        game, norm = BaseEnv.get_game_norm(exp_setup)
        return LLMEnv(
            num_agents=exp_setup.num_agents,
            num_rounds=exp_setup.num_rounds,
            game=game,
            norm=norm,
            ps=ps,
            setup_assesment=setup_assesment,
            step_assesment=step_assesment,
            complete_assesment=complete_assesment
        )
