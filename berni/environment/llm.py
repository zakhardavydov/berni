import os
import time
import json
import numpy as np
import threading
import matplotlib
import matplotlib.pyplot as plt

from nypd.environment import BaseEnv
from nypd.game import AbsGame
from nypd.norms import AbsNorm
from nypd.ps import AbsPartnerSelection
from nypd.structures import ExperimentSetup, Action
from nypd.agent import AbsAgent

from berni.assesment import AgentAssesor


matplotlib.use('Agg')

llm_env_call_lock = threading.Lock()


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

    def _send_batch(self, llm, prompts: list[str]) -> list[str]:
        with llm_env_call_lock:
            return llm.batch(prompts)

    def _step(self):
        self.rounds += 1
        # Removing structures and rewards of the previous round
        self.actions = []
        self.rewards = np.zeros(self.num_agents)
        self.state = [[] for _ in range(self.num_agents)]

        self.pairs = self.ps.select(self.pairs, 0, self.num_agents)
        self.prev_round_scores = self.scores.copy()

        agent_prompts = {}
        
        for i, j in self.pairs:
            agent1: AbsAgent = self.agents[i]
            agent2: AbsAgent = self.agents[j]

            bias_gap = agent2.bias_score - agent1.bias_score

            agent_prompts[i] = agent1.preplay(j)
            agent_prompts[j] = agent2.preplay(i)

        # Implement smarter batching, looking at agent strategy LLM and only batching to same underlying LLM
        llm = self.agents[0].strategy._llm

        filtered_prompts = {k: v for k, v in agent_prompts.items() if v is not None}

        responses = self._send_batch(llm, list(filtered_prompts.values()))

        keys = list(filtered_prompts.keys())

        agent_response = {keys[i]: responses[i] for i in range(len(responses))}

        for i, j in self.pairs:
            agent1: AbsAgent = self.agents[i]
            agent2: AbsAgent = self.agents[j]
            
            if not i in agent_response:
                act1 = Action.D
            else:
                act1 = agent1.postplay(agent_response[i], j)

            if not j in agent_response:
                act2 = Action.D
            else:
                act2 = agent1.postplay(agent_response[j], i)

            payoff = self.game.get_payoff(action=(act1, act2))
            self.current_payoffs.clear()
            self.current_payoffs[i] = payoff[0]
            self.current_payoffs[j] = payoff[1]

            w1 = self.norm.calculate_reward(agent1, agent2)
            w2 = self.norm.calculate_reward(agent2, agent1)

            r1 = payoff[0] + w1
            r2 = payoff[1] + w2

            self.actions.append((act1, act2))

            self.state[i].append((j, (act1, act2)))
            self.state[j].append((i, (act2, act1)))

            self.rewards[i] += r1
            self.rewards[j] += r2

            agent1.save_history(
                opponent=j,
                opponent_model=agent1,
                action=act1,
                reward=r1,
                round=self.rounds,
                bias_gap=bias_gap
            )
            agent2.save_history(
                opponent=i,
                opponent_model=agent2,
                action=act2,
                reward=r2,
                round=self.rounds,
                bias_gap=bias_gap
            )

        # Updating the agent rewards in their objects
        for i in range(self.num_agents):
            self.agents[i].observe(self.rewards[i], self.state[i])

        # Updating total score and checking if the game is done
        self.scores += self.rewards

        if self.rounds >= self.num_rounds:
            self.done = True

    def step(self):
        self._step()

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
        if self.results_dir:
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
        """
        self._plot_flips(run)
        self._plot_bias(run)
        """
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
