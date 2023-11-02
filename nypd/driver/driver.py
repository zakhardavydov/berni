import os
import logging

from tqdm import tqdm
from typing import Any, List, Optional, Tuple, Type, Dict

import mlflow

from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.store.entities import PagedList

import nypd.stats as stats

from nypd.env import AbsEnv, BaseEnv
from nypd.core import settings
from nypd.game import game_registry
from nypd.norms import norm_registry
from nypd.benchmark import AbsBenchmark
from nypd.strategy.q_table import *
from nypd.structures import (
    AgentConfigs,
    AgentConfigPrePlay,
    AgentConfig,
    RunInfo
)

from .artifact_processer import *


class Driver:

    def __init__(self):
        pass

    @staticmethod
    def get_exp_name(name: str) -> str:
         return f"{settings.CURRENT_USER_DIR}/{name}"

    @staticmethod
    def concate_runs_df(
            runs: Dict[str, Any],
            y_col: str = "coop",
            sample_rate: int = 100
    ) -> pd.DataFrame:
        df = pd.concat([run[1].get_df(y_col) for run in runs.values()], axis=0)
        frames = np.random.randint(0, df["round"].max(), sample_rate).tolist()
        sampled = df.loc[df["round"].isin(frames)]
        return sampled

    @staticmethod
    def concate_runs_df_on_info(
            runs: Dict[str, RunInfo],
            y_col: str,
            sample_rate: int = 100
    ) -> pd.DataFrame:
        df = pd.concat([run.collector.get_df(y_col) for run in runs.values()], axis=0)
        frames = np.random.randint(0, df["round"].max(), sample_rate).tolist()
        sampled = df.loc[df["round"].isin(frames)]
        return sampled

    @staticmethod
    def generate_run_benchmark_df(runs: Dict[str, RunInfo],
                                  benchmark_performance: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        out = []
        for run_id, run in runs.items():
            base = run.exp_info.plottable_params()
            perf = benchmark_performance[run_id]
            out.append(
                {**base, **perf}
            )
        return pd.DataFrame(out)

    @staticmethod
    def log_current_artifact(
        client: MlflowClient,
        run: Run,
        collector: StatsCollector,
        exp_setting: ExperimentSetup
    ):
        artifact_processor = ArtifactProcessor(collector, exp_setting)
        # log data, artifact
        if not os.path.exists(f"nypd/artifacts/{run.info.run_id}"):
            os.makedirs(f"nypd/artifacts/{run.info.run_id}")
        artifact_processor.to_csv(f"nypd/artifacts/{run.info.run_id}/current_run_info.csv")
        artifact_processor.pkl_collector(f"nypd/artifacts/{run.info.run_id}/current_run_info.pkl")
        artifact_processor.save_plot(f"nypd/artifacts/{run.info.run_id}/current_run_info.png")
        artifact_processor.save_html(f"nypd/artifacts/{run.info.run_id}/current_run_info.html")
        artifact_processor.pkl_exp_info(f"nypd/artifacts/{run.info.run_id}/current_run_exp_info.pkl")
        artifact_processor.save_exp_info_json(f"nypd/artifacts/{run.info.run_id}/current_run_exp_info.json")
        
        local_artifact_path = "nypd/artifacts/{}".format(run.info.run_id)
        client.log_artifacts(run.info.run_id, local_dir=local_artifact_path)

    @staticmethod
    def fetch_data_from_runs(runs: PagedList[Run], dst_path: str):
        # cus of static method, need to configure mlflow under
        mlflow.set_tracking_uri(settings.ML_FLOW_URL)
        mlflow.set_registry_uri(settings.ML_FLOW_URL)

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        # NOTE: NEED FULL ACCESS PERMISSION TO DISK, WHICH I THINK IT'S DANGEROUS HERE
        # else:
        #     # remove all existing files
        #     filelist = glob.glob(os.path.join(dst_path, "**"))
        #     for f in filelist:
        #         os.remove(f)

        logging.info(f"Downloading experiment artifacts into: {dst_path}")
        for idx, cur_run in tqdm(enumerate(runs), total=len(runs)):
            cur_info = cur_run.info
            cur_run_id = cur_info.run_id
            cur_dst_path = os.path.join(dst_path, f"{idx}_{cur_run_id}")
            if not os.path.exists(cur_dst_path):
                os.mkdir(cur_dst_path)
            if os.path.exists(os.path.join(cur_dst_path, "current_run_info.pkl")) and os.path.exists(os.path.join(cur_dst_path, "current_run_exp_info.json")):
                continue
            else:
                mlflow.artifacts.download_artifacts(run_id=cur_run_id, artifact_path="current_run_info.pkl", dst_path=cur_dst_path)
                mlflow.artifacts.download_artifacts(run_id=cur_run_id, artifact_path="current_run_exp_info.json", dst_path=cur_dst_path)

    @staticmethod
    def load_local_runs(dst_path: str) -> Dict[str, Tuple[str, StatsCollector]]:
        runs_file_info = sorted(os.listdir(dst_path), key=lambda cur_str: int(cur_str.split("_")[0]))

        runs_dict = {}
        for cur_run_id in runs_file_info:
            cur_dst_path = os.path.join(dst_path, f"{cur_run_id}")
            with open(os.path.join(cur_dst_path, "current_run_exp_info.json"), "rb") as f:
                cur_run_exp_info = json.load(f)
                new = json.loads(cur_run_exp_info)
                for agent_type in new["config"]:
                    t = agent_type["config"]["type"]
                    r = agent_type["raio"]
                    new[f"{t}-ratio"] = r
                cur_run_exp_info = json.dumps(new)
            with open(os.path.join(cur_dst_path, "current_run_info.pkl"), "rb") as f:
                cur_run_info = pkl.load(f)
                run_info = json.loads(cur_run_exp_info)
                cur_run_info.stat_df = cur_run_info._init_df(
                    cur_run_info.metrics, run_id=cur_run_id, params=run_info["config"]
                )
            runs_dict[cur_run_id] = (cur_run_exp_info, cur_run_info)
        return runs_dict

    @staticmethod
    def get_run_info_structs(runs: Dict[str, Tuple[str, StatsCollector]]) -> Dict[str, RunInfo]:

        def __gen_exp_setup(run_info):
            loaded = json.loads(run_info[0])
            for config in loaded["config"]:
                if "raio" in config:
                    config["ratio"] = config["raio"]
            loaded["config"] = {"configs": loaded["config"]}
            return ExperimentSetup(**loaded)

        return {
            run_id: RunInfo(
                exp_info=__gen_exp_setup(run_info),
                collector=run_info[1]
            )
            for run_id, run_info in runs.items()
        }

    @staticmethod
    def download_analysis_data_from_server(experiment_id: int, override_local: bool = False) -> Dict[str, Tuple[str, StatsCollector]]:
        # cus of static method, need to configure mlflow under
        dst_path = f"nypd/data_from_mlflow/exp_id_{experiment_id}"

        if not os.path.exists(dst_path) or override_local:
            mlflow.set_tracking_uri(settings.ML_FLOW_URL)
            mlflow.set_registry_uri(settings.ML_FLOW_URL)

            client = mlflow.MlflowClient(tracking_uri=settings.ML_FLOW_URL, registry_uri=settings.ML_FLOW_URL)
            runs = client.search_runs(experiment_ids=experiment_id)

            Driver.fetch_data_from_runs(runs, dst_path)
        else:
            logging.info(f"Fetched local artifact dir: {dst_path}")
        return Driver.load_local_runs(dst_path)

    @staticmethod
    def apply_benchmark_on_run(run: RunInfo, benchmarks: Dict[str, AbsBenchmark]) -> Dict[str, float]:
        return {name: benchmark.apply(run.collector) for name, benchmark in benchmarks.items()}

    @staticmethod
    def apply_benchmark_on_runs(
            runs: Dict[str, RunInfo],
            benchmarks: Dict[str, AbsBenchmark]
    ) -> Dict[str, Dict[str, float]]:
        return {
            run_id: Driver.apply_benchmark_on_run(run, benchmarks)
            for run_id, run in runs.items()
        }

    @staticmethod
    def _play_with_setup_aux(
            exp: ExperimentSetup,
            exp_name: str,
            ml_flow_client: Optional[mlflow.MlflowClient] = None,
            tracked_live: bool = False
    ) -> Tuple[StatsCollector, Run]:
        game = game_registry.registry[exp.game](**exp.game_params)
        norm = norm_registry.registry[exp.norm](**exp.norm_params)

        env = BaseEnv(exp.num_agents, exp.num_rounds, game=game, norm=norm)

        agents, st_ratio = registry.seed(
            env=env,
            agents=exp.config,
            count=exp.num_agents,
        )

        experiment = ml_flow_client.get_experiment_by_name(exp_name)
        if experiment is None:
            ml_flow_client.create_experiment(exp_name)
            experiment = ml_flow_client.get_experiment_by_name(exp_name)

        run = ml_flow_client.create_run(experiment_id=experiment.experiment_id)
        run = mlflow.start_run(run_id=run.info.run_id)
        ratio = stats.RatioPie(st_ratio, ml_flow_client, run.info.run_id)

        # NOTE: ADD PARAM that want to logged in MLflow here!!!
        ml_flow_client.log_param(run.info.run_id, "num-agents", exp.num_agents)
        ml_flow_client.log_param(run.info.run_id, "num-rounds", exp.num_rounds)
        ml_flow_client.log_param(run.info.run_id, "active-norm", norm.name)
        ml_flow_client.log_param(run.info.run_id, "norm-reward", norm.reward)

        collected = Driver.play_predefined_agents(env, agents, ml_flow_client, run.info.run_id, tracked_live=tracked_live)
        collected.ratio_tracker = ratio # maybe can be combined with next line, kept seperate just to not break the code
        collected.add_metric("ratio", ratio)
        
        # NOTE: ADD artifact want ot logged in MLflow here!!!
        Driver.log_current_artifact(ml_flow_client, run, collected, exp)

        mlflow.end_run()
        return collected, run
    
    @staticmethod
    def play_with_setup(
            exp: ExperimentSetup,
            exp_name: str,
            ml_flow_client: Optional[mlflow.MlflowClient] = None,
            initial_seed: int = 42,
            number_of_times: int = 1,
            tracked_live: bool = False
    ) -> None:
        # seed numpy
        np.random.seed(initial_seed)
        seeds = np.random.randint(np.iinfo(np.int16).max, size=number_of_times)

        logging.info(f"Seeds generated -> {seeds}")

        for cur_seed in seeds:
            np.random.seed(cur_seed)
            random.seed(int(cur_seed))
            Driver._play_with_setup_aux(
                exp,
                exp_name,
                ml_flow_client,
                tracked_live=tracked_live
            )


    @staticmethod
    def play_predefined_agents(
            env: AbsEnv,
            agents: List[AbsAgent],
            ml_flow_client: Optional[mlflow.MlflowClient] = None,
            run_id: Optional[str] = None,
            tracked_live: bool = False
    ) -> stats.StatsCollector:
        env.reset()

        hcr = stats.HCR(env.num_rounds)
        avg = stats.AvScore(env.num_rounds)
        coop = stats.CoopRate(env.num_rounds)

        for agent in agents:
            agent.clean()
            env.add_agent(agent)

        for _ in tqdm(range(env.num_rounds)):
            env.step()

            hcr.add_sample(env.scores, ml_flow_client, run_id, tracked_live)
            avg.add_sample(env.rewards, ml_flow_client, run_id, tracked_live)
            coop.add_sample(env.actions, ml_flow_client, run_id, tracked_live)

        return stats.StatsCollector(
            metrics={
                "hcr": hcr,
                "avg": avg,
                "coop": coop
            }
        )

    @staticmethod
    def variate_proportion(
            num_agents: int,
            num_rounds: int,
            exp_name,
            av_configs: List[AgentConfig],
            av_norm: List[str],
            av_game: List[str],
            variation_step: float = 0.1,
            ml_flow_client: Optional[mlflow.MlflowClient] = None,
            initial_seed: int = 42,
            number_of_times: int = 1,
            tracked_live: bool = False
    ):
        np.random.seed(initial_seed)
        seeds = np.random.randint(np.iinfo(np.int16).max, size=number_of_times)
        logging.info(f"Seeds generated -> {seeds}")

        for cur_seed in seeds:
            np.random.seed(cur_seed)
            random.seed(cur_seed)
            Driver._variate_proportion_aux(
                num_agents=num_agents,
                num_rounds=num_rounds,
                exp_name=exp_name,
                av_configs=av_configs,
                av_norm=av_norm,
                av_game=av_game,
                variation_step=variation_step,
                ml_flow_client=ml_flow_client,
                tracked_live=tracked_live
            )

    @staticmethod
    def _variate_proportion_aux(
            num_agents: int,
            num_rounds: int,
            exp_name,
            av_configs: List[AgentConfig],
            av_norm: List[str],
            av_game: List[str],
            variation_step: float = 0.1,
            ml_flow_client: Optional[mlflow.MlflowClient] = None,
            tracked_live: bool = False
    ):
        configs: List[AgentConfigs] = []
        for prop in np.arange(0.0, 1.0, variation_step):
            first_config = av_configs[0]
            against_config = av_configs[1]
            configs.append(
                AgentConfigs(
                    configs=[
                        AgentConfigPrePlay(
                            config=first_config,
                            ratio=prop
                        ),
                        AgentConfigPrePlay(
                            config=against_config,
                            ratio=1 - prop
                        )
                    ]
                )
            )
        setups = []
        for norm in av_norm:
            for game in av_game:
                for config in configs:
                    setups.append(
                        ExperimentSetup(
                            num_agents=num_agents,
                            num_rounds=num_rounds,
                            game=game,
                            norm=norm,
                            config=config
                        )
                    )
        for setup in tqdm(setups):
            Driver.play_with_setup(setup, exp_name, ml_flow_client, tracked_live=tracked_live)
