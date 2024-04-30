import os
import ast
import json
import numpy as np
import pandas as pd
import colorcet as cc
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


class BaseGameInvestigator:

    def __init__(
            self,
            game_path: str,
            party: float,
            opinion_group_range: tuple[float, float] | None = None,
            opinion_embeddings_model: str = "thenlper/gte-large",
            plot_experiment_override: bool = True,
            plot_dir: str = "plots",
            seed: int = 42
    ) -> None:

        if not opinion_group_range:
            opinion_group_range = (0, 1)
        
        self._game_path = game_path
        self._plot_dir = plot_dir

        self._plot_experiment_override = plot_experiment_override
        self._experiment_dir = self._game_path.strip(os.sep).split(os.sep)[1]
        self._experiment_name, self._experiment_id = self._experiment_dir.split("--")
        
        self._plot_path = os.path.join(self._plot_dir, self._experiment_name if self._plot_experiment_override else self.__experiment_dir)
        if not os.path.exists(self._plot_path):
            os.makedirs(self._plot_path)

        self._party = party
        self._opinion_group_range = opinion_group_range
        self._og = f"opinion_group__{self._party}"

        self._df = self._stich_dir_for_exp()
        self._unique_bias = {}
        
        self._opinion_embeddings_model = opinion_embeddings_model
        self._seed = seed
        
        self.opinion_index = None
        self.reverse_opinion_index = None
        
        self.opinion_vector_index = None
        self.opinion_vector_cluster_index = None

    def process_prefix(self, optimized_df: pd.DataFrame, prefix: str, override: bool = False):
        optimized_df = optimized_df.copy()

        ratio_df = self.calculate_ratio(optimized_df)

        self.opinion_dynamics_viz(ratio_df, prefix)
        self.plot_ratio_grid(ratio_df, prefix)

        flips_df = self.flips_df(optimized_df, override_cache=override)
        self.flips_bar_viz(flips_df, prefix=prefix)

        flips_sim_df = self.flips_similarity(optimized_df, override_cache=override)
        self.flips_similarity_viz(flips_sim_df, prefix=prefix)

        variance_analysis_path = os.path.join(self._plot_path, "variance")
        if not os.path.exists(variance_analysis_path):
            os.makedirs(variance_analysis_path)

        self.round_variance_prompt_structure_viz(flips_sim_df, distance="outcome", prefix=prefix)
        self.round_variance_all_viz(flips_sim_df, distance="outcome", prefix=prefix)
        self.round_variance_all_viz(flips_sim_df, distance="outcome", is_bar=True, prefix=prefix)
        self.round_variance_prompt_structure_viz(flips_sim_df, distance="opponent", prefix=prefix)
        self.round_variance_all_viz(flips_sim_df, distance="opponent", prefix=prefix)
        self.round_variance_all_viz(flips_sim_df, distance="opponent", is_bar=True, prefix=prefix)

    def process(self, override: bool = False) -> pd.DataFrame:
        optimized_df = self.get_optimized_opinion_df(override)
        self.all_opinion_vector_viz(optimized_df)
        self.process_prefix(optimized_df, prefix="bias_dynamics")
        cluster_substituted_df = self.substitute_bias_score_on_cluster(optimized_df)
        self.process_prefix(cluster_substituted_df, prefix="cluster_dynamics")
        return optimized_df

    def substitute_bias_score_on_cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(
            subset=["outcome_opinion", "my_prev_opinion", "my_opponent_prev_opinion"]
        )
        df = df.astype({"outcome_opinion": int, "my_prev_opinion": int, "my_opponent_prev_opinion": int})

        print(len(self.opinion_vector_cluster_index))
        print(len(df["outcome_opinion"].unique()))

        def map_to_cluster(df: pd.DataFrame, source_col: str, target_col: str):

            def row_mapper(row):
                row[target_col] = self.opinion_vector_cluster_index[f"{row['prompt_structure']}__{row[source_col]}"]
                return row
            
            return df.apply(row_mapper, axis=1)
        
        df = map_to_cluster(df, "outcome_opinion", "bias_score")
        df = map_to_cluster(df, "my_prev_opinion", "my_prev_bias_score")
        df = map_to_cluster(df, "my_opponent_prev_opinion", "opponent_bias_before_round")

        return df

    def _stich_dir_for_exp(self) -> pd.DataFrame:
        batch_dfs = []

        for batch_id in os.listdir(self._game_path):
            batch_path = os.path.join(self._game_path, batch_id)
            if not os.path.isdir(batch_path):
                continue

            dfs = []
            for filename in os.listdir(batch_path):
                file_path = os.path.join(batch_path, filename)
                if os.path.isfile(file_path) and filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    df = df[(df[self._og] >= self._opinion_group_range[0]) & (df[self._og] <= self._opinion_group_range[1])]
                    dfs.append(df)

            if dfs:
                big_df = pd.concat(dfs, ignore_index=True)
                batch_dfs.append(big_df)

        combined_df = pd.concat(batch_dfs, ignore_index=True)

        return combined_df
    
    def opinion_dynamics_viz(self, df: pd.DataFrame, prefix: str):
        palette = sns.color_palette("viridis")
        sns.set_theme(style="darkgrid")
        g = sns.FacetGrid(df, row="prompt_structure", col="grid_size", palette=palette)
        g.map_dataframe(sns.lineplot, x='round', y='bias_av', hue=self._og, estimator='mean', ci='sd', palette=palette)
        g.add_legend()
        g.set_titles(row_template='{row_name}', col_template='{col_name}', fontsize=8)
        g.set_axis_labels('Round', 'Average Bias')

        plt.savefig(f"{self._plot_path}/{prefix}_opinion_dynamics_{self._party}_multi.png")

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def batch_vectorize(self, texts: str, batch_size: int = 32, override_cache: bool = False) -> list[list[float]]:
        agent_results_indexed_path = os.path.join(self._game_path, "agent_results_indexed.pt")

        if os.path.exists(agent_results_indexed_path) and not override_cache:
            vectors = torch.load(agent_results_indexed_path)
            return vectors.tolist()

        tokenizer = AutoTokenizer.from_pretrained(self._opinion_embeddings_model)
        model = AutoModel.from_pretrained(self._opinion_embeddings_model, device_map="cuda")
    
        gpu = torch.device("cuda")
        cpu = torch.device("cpu")

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_dict = tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")

            batch_dict = batch_dict.to(gpu)

            outputs = model(**batch_dict)
            embeddings = self._average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

            embeddings = F.normalize(embeddings, p=2, dim=1)

            embeddings = embeddings.to(cpu)
            embeddings = embeddings.tolist()

            all_embeddings.extend(embeddings)
        
        all_embeddings_tensor = torch.Tensor(all_embeddings)
        torch.save(all_embeddings_tensor, agent_results_indexed_path)
        
        return all_embeddings

    def _build_reverse_opinion_index(self):
        self.reverse_opinion_index = {opinion: int(index) for index, opinion in self.opinion_index.items()}

    def _build_opinion_index(self, opinion_index):
        self.opinion_index = {int(opinion_id): opinion for opinion_id, opinion in opinion_index.items()}

    def _build_opinion_vector_index(self, opinion_index):
        self.opinion_vector_index = {int(opinion_id): opinion for opinion_id, opinion in opinion_index.items()}

    def get_opinion_index(self, raw_agent_df: pd.DataFrame | None = None, override_cache: bool = False) -> dict[int, str]:
        index_path = os.path.join(self._game_path, "opinion_index.json")
        if override_cache and raw_agent_df is None:
            raise ValueError("Cannot override cache as raw_agent_df is None")
        if not override_cache and os.path.exists(index_path):
            with open(index_path, "r") as f:
                opinion_index = json.load(f)
                self._build_opinion_index(opinion_index)
                self._build_reverse_opinion_index()
                return opinion_index

        outcome_opinions = raw_agent_df['outcome_opinion'].tolist()
        unique_opinions = np.unique(outcome_opinions).tolist()
        opinion_index = {i: unique_opinion for i, unique_opinion in enumerate(unique_opinions)}
        with open(index_path, "w") as f:
            json.dump(opinion_index, f)
        self._build_opinion_index(opinion_index)
        self._build_reverse_opinion_index()
        return opinion_index

    def get_vector_opinion_index(self, override_cache: bool = False) -> dict[int, list[float]]:

        vector_index_path = os.path.join(self._game_path, "vector_opinion_index.json")
        if os.path.exists(vector_index_path) and not override_cache:
            with open(vector_index_path, "r") as f:
                self._build_opinion_vector_index(json.load(f))
                return self.opinion_vector_index

        assert self.opinion_index, "Opinion index is needed to rebuild opinion vector index"

        unique_opinions = list(self.opinion_index.values())
        
        opinion_vectors = self.batch_vectorize(unique_opinions, override_cache=override_cache)
        self.opinion_vector_index = {
            index: unique_opinion_vector
            for index, unique_opinion_vector in enumerate(opinion_vectors)
        }

        with open(vector_index_path, "w") as f:
            json.dump(self.opinion_vector_index, f)
        
        return self.opinion_vector_index
    
    def get_optimized_opinion_df(self, override_cache: bool = False) -> pd.DataFrame:

        optimized_path = os.path.join(self._game_path, "agent_optimized.csv")
        
        if os.path.exists(optimized_path) and not override_cache:
            self.get_opinion_index(None)
            self.get_vector_opinion_index(None)
            return pd.read_csv(optimized_path)

        def map_list(column):
            if column is None:
                return None
            try:
                return self.reverse_opinion_index.get(column[0])
            except Exception as e:
                print(f"Failed at {column} with exception: {e}")
            return None
        
        full_df = self.all_opinion_distribution(override_cache)
        
        self.get_opinion_index(full_df, override_cache)
        self.get_vector_opinion_index(override_cache)

        full_df["outcome_opinion"] = full_df["outcome_opinion"].map(self.reverse_opinion_index)
        full_df["my_prev_opinion"] = full_df["my_prev_opinion"].map(self.reverse_opinion_index)
        full_df["my_opponent_prev_opinion"] = full_df["my_opponent_prev_opinion"].map(self.reverse_opinion_index)
        full_df["round_neighbours_opinion"] = full_df["round_neighbours_opinion"].apply(map_list)

        full_df = full_df.drop(columns=["round_prompt"])

        full_df.to_csv(optimized_path)

        return full_df

    def index_simulation(self, batch_id: str, sim_id: str) -> pd.DataFrame:
        sim_path = os.path.join(self._game_path, batch_id, sim_id, "matrix")
        full_df = pd.DataFrame()

        prev = None

        for filename in os.listdir(sim_path):
            if filename.endswith('.json'):
                file_path = os.path.join(sim_path, filename)
                round_number = int(filename.split('.')[0])

                with open(file_path, 'r') as file:
                    data = json.load(file)

                    agent_rows = []
                    for agent_id, agent_data in data.items():
                        agent_data["agent_id"] = int(agent_id)
                        agent_data["round"] = round_number
                        if prev:
                            opponent_id = agent_data["round_neighbours"][0]
                            
                            opponent_bias_before_round = prev[str(opponent_id)]["bias_score"]
                            my_prev_opinion = prev[agent_id]["outcome_opinion"]
                            my_prev_bias_score = prev[agent_id]["bias_score"]
                            my_opponent_prev_opinion = prev[str(opponent_id)]["outcome_opinion"]

                            agent_data["opponent_bias_before_round"] = opponent_bias_before_round
                            agent_data["my_prev_opinion"] = my_prev_opinion
                            agent_data["my_prev_bias_score"] = my_prev_bias_score
                            agent_data["my_opponent_prev_opinion"] = my_opponent_prev_opinion
                            
                        agent_rows.append(agent_data)
                        
                    temp_df = pd.DataFrame.from_records(agent_rows)
                    prev = data

                    bias_ratios = {key: 0 for key in self._unique_bias.keys()}

                    for _, agent_data in data.items():
                        bias = agent_data["bias_score"]

                        self._unique_bias[bias] = 0

                        if bias not in bias_ratios:
                            bias_ratios[agent_data["bias_score"]] = 1
                        else:
                            bias_ratios[agent_data["bias_score"]] += 1

                    ratios = [{f"bias_ratio": val / len(data), "bias_score": bias} for bias, val in bias_ratios.items()]
                    temp_ratio_df = pd.DataFrame.from_records(ratios)
                    temp_ratio_df["round"] = round_number
                    
                    full_df = pd.concat([full_df, temp_df], ignore_index=True)
        
        return full_df

    def calculate_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby(['prompt_structure', 'grid_size', self._og, 'round'])

        def calculate_proportions(group):
            count_total_agents = group['agent_id'].count()
            proportions = group['bias_score'].value_counts().div(count_total_agents).reset_index()
            proportions.columns = ['bias_score', 'bias_ratio']
            return proportions

        proportions_list = []
        for name, group in grouped:
            props = calculate_proportions(group)
            for col, val in zip(['prompt_structure', 'grid_size', self._og, 'round'], name):
                props[col] = val
            proportions_list.append(props)

        result_df = pd.concat(proportions_list).reset_index(drop=True)

        weighted_averages = result_df.groupby(['prompt_structure', 'grid_size', 'opinion_group__-1', 'round']).apply(
            lambda x: (x['bias_score'] * x['bias_ratio']).sum() / x['bias_ratio'].sum()
        ).reset_index(name='bias_av')

        result_df = pd.merge(result_df, weighted_averages, on=['prompt_structure', 'grid_size', 'opinion_group__-1', 'round'])

        return result_df

    def all_opinion_distribution(self, override_cache: bool = False, cache_raw: bool = False) -> pd.DataFrame:
        full_agent_results_path = os.path.join(self._game_path, "raw_agent_results.csv")

        if os.path.exists(full_agent_results_path) and not override_cache:
            return pd.read_csv(full_agent_results_path)
        
        full_results = []
        ratio_results = []

        for idx, group in self._df.groupby(["batch_id", "grid_size", "prompt_structure", self._og]):
            batch_id, grid_size, prompt_structure, opinion_group = idx
            print("INDEXING COMBINATION")
            print(batch_id, grid_size, prompt_structure, opinion_group)
            filtered_sim_values = self._df[
                (self._df["prompt_structure"] == prompt_structure) &
                (self._df["grid_size"] == grid_size) &
                (self._df[self._og] == opinion_group)
            ]["sim_id"].unique().tolist()

            seeds_full = [self.index_simulation(batch_id, sim_id) for sim_id in filtered_sim_values]

            concated_full_df = pd.concat(seeds_full, ignore_index=True)
            concated_full_df["grid_size"] = grid_size
            concated_full_df["prompt_structure"] = prompt_structure
            concated_full_df[self._og] = opinion_group

            full_results.append(concated_full_df)

        full_results_df = pd.concat(full_results, ignore_index=True)
        
        if cache_raw:
            full_results_df.to_csv(full_agent_results_path)

        return full_results_df
        
    def plot_ratio_grid(self, df: pd.DataFrame, prefix):
        df = df.groupby(['bias_score', 'round', 'prompt_structure', 'opinion_group__-1'])['bias_ratio'].mean().reset_index()

        p = so.Plot(data=df, x="round", y="bias_ratio", color="bias_score")
        p = p.add(so.Area(alpha=.7), so.Stack())
        p = p.facet("prompt_structure", self._og)
        p = p.scale(color="viridis").layout(engine="constrained")

        p.save(os.path.join(self._plot_path, f"{prefix}_dynamics_ratio_profile.png"), bbox_inches="tight")

    def flips_df(self, df: pd.DataFrame, normalize_by_agent_count: bool = False, override_cache: bool = False):

        df = df.dropna(
            subset=["outcome_opinion", "my_prev_opinion", "my_opponent_prev_opinion", "opponent_bias_before_round", "my_prev_bias_score", "bias_score"]
        )

        results = []
        grouped = df.groupby(['grid_size', 'prompt_structure', self._og])
        
        for name, group in grouped:
            agent_count = name[0] * name[0] if normalize_by_agent_count else 1
            group_copy = group.copy()

            group_copy['opinion_flip'] = (group_copy['bias_score'] != group_copy['my_prev_bias_score'])

            group_copy['flip_type'] = group_copy.apply(
                lambda row: f"{int(row['bias_score'])}__{int(row['opponent_bias_before_round'])}"
                if row['opinion_flip'] else None, axis=1
            )
            
            flips_df = group_copy[group_copy['opinion_flip']].copy()
            pivot = flips_df.pivot_table(index='round', columns='flip_type', aggfunc='size', fill_value=0)
            normalized_pivot = pivot / agent_count

            multi_index = pd.MultiIndex.from_product([[name[0]], [name[1]], [name[2]], pivot.index],
                                                        names=['Grid Size', 'Prompt Structure', 'Opinion Group', 'Round'])
            normalized_pivot.index = multi_index
            results.append(normalized_pivot)

        final_result = pd.concat(results, axis=0).fillna(0)
        return final_result

    def uncomress_vector_index(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_to_replace = ['outcome_opinion', 'my_prev_opinion', 'my_opponent_prev_opinion']

        for column in columns_to_replace:
            df[f"{column}__uncompressed"] = df[column].map(self.opinion_vector_index)
        
        return df

    def flips_similarity(self, agent_vector_df: pd.DataFrame, override_cache: bool = False):

        data_cleaned = agent_vector_df.dropna(subset=["outcome_opinion", "my_prev_opinion", "my_opponent_prev_opinion"])
        
        data_cleaned = self.uncomress_vector_index(data_cleaned)

        def calculate_distances(row):
            outcome = np.array(row["outcome_opinion__uncompressed"]).reshape(1, -1)
            my_prev = np.array(row["my_prev_opinion__uncompressed"]).reshape(1, -1)
            my_opponent_prev = np.array(row["my_opponent_prev_opinion__uncompressed"]).reshape(1, -1)

            opponent_similarity = cosine_similarity(my_prev, my_opponent_prev)
            outcome_similarity = cosine_similarity(outcome, my_prev)

            row["opponent_similarity"] = opponent_similarity[0][0]
            row["outcome_similarity"] = outcome_similarity[0][0]
            
            return row
        
        data_cleaned = data_cleaned.apply(calculate_distances, axis=1)

        return data_cleaned

    def flips_similarity_viz(self, df: pd.DataFrame, prefix: str, bins=25):

        def create_heatmap_data(df):
            filtered_df = df[(df['opponent_similarity'] >= 0.75) & (df['outcome_similarity'] >= 0.75)]
            heatmap_data, x_edges, y_edges = np.histogram2d(
                filtered_df['opponent_similarity'], filtered_df['outcome_similarity'],
                bins=bins, range=[[0.75, 1], [0.75, 1]]
            )
            heatmap_df = pd.DataFrame(heatmap_data)
            heatmap_df.columns = [f"{round(y_edges[j], 2)}-{round(y_edges[j+1], 2)}" for j in range(len(y_edges)-1)]
            heatmap_df.index = [f"{round(x_edges[i], 2)}-{round(x_edges[i+1], 2)}" for i in range(len(x_edges)-1)]
            return heatmap_df

        def draw_heatmap(*args, **kwargs):
            df = kwargs["data"]
            data = create_heatmap_data(df)
            ax = plt.gca()
            sns.heatmap(data, cmap='viridis', ax=ax, cbar=True)
            
            ax.set_xlabel("Pre-round between agent/opponent")
            ax.set_ylabel("Pre-round/post-round")

        g = sns.FacetGrid(df, col='prompt_structure', height=5, aspect=1, margin_titles=True)
        g.map_dataframe(draw_heatmap, 'prompt_structure')

        g.savefig(os.path.join(self._plot_path, f"{prefix}_flips_similarity_heatmap.png"))

    def round_variance_prompt_structure_viz(self, data: pd.DataFrame, prefix: str, distance: str = "outcome"):
        diversity_by_round_and_prompt = data.groupby(['round', 'prompt_structure'])[f'{distance}_similarity'].var().reset_index()

        plt.figure(figsize=(16, 8))
        sns.barplot(data=diversity_by_round_and_prompt, x='round', y=f'{distance}_similarity', hue='prompt_structure', palette='viridis')

        plt.title(f'Diversity of {distance} similarity by round and prompt structure')
        plt.xlabel('Round')
        plt.ylabel(f'Variance of {distance} similarity')
        plt.legend(title='Prompt Structure')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()

        plt.savefig(os.path.join(self._plot_path, "variance", f"{prefix}_round_{distance}_prompt_structure_variance.png"))

    def round_variance_all_viz(self, data: pd.DataFrame, prefix: str, distance: str = "outcome", is_bar: bool = False):
        diversity_data = data.groupby(['round', 'prompt_structure', 'grid_size', self._og])[f'{distance}_similarity'].var().reset_index()
        
        diversity_data['group'] = diversity_data['prompt_structure'].astype(str) + '_' + diversity_data['grid_size'].astype(str) + '_' + diversity_data[self._og].astype(str)
        diversity_data['group_smaller'] = diversity_data['prompt_structure'].astype(str)

        plt.figure(figsize=(24, 6))
        if is_bar:
            sns.barplot(data=diversity_data, x='round', y=f'{distance}_similarity', hue='group', palette='viridis', width=1)
        else:
            sns.lineplot(data=diversity_data, x='round', y=f'{distance}_similarity', hue='group_smaller', palette='viridis', marker='o', err_style='band', ci='sd')

        # Set the plot title and labels
        plt.title(f'Diversity of {distance} similarity by round, prompt structure, grid size, and opinion group')
        plt.xlabel('Round')
        plt.ylabel(f'Variance of {distance} similarity')
        plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()

        plt.savefig(os.path.join(self._plot_path, "variance", f"{prefix}_round_{distance}_all_variance_{'bar' if is_bar else 'line'}.png"))

    def flips_bar_viz(self, data: pd.DataFrame, prefix: str):
        flip_columns = ['-1__0', '-1__1', '0__-1', '0__1', '1__-1', '1__0']
    
        grouped_data = data.groupby(['Prompt Structure', 'Grid Size', 'Opinion Group'])[flip_columns].sum()
        grouped_data.reset_index(inplace=True)

        unique_combinations = grouped_data['Opinion Group'].astype(str) + ' - ' + grouped_data['Grid Size'].astype(str)
        grouped_data['Opinion_Grid'] = pd.Categorical(unique_combinations, categories=unique_combinations.unique(), ordered=True)
        
        colors = sns.color_palette('viridis', len(flip_columns))
        
        fig, ax = plt.subplots(figsize=(20, 10))
        width = 0.8 / len(grouped_data['Prompt Structure'].unique())

        previous_grid_size = None
        
        for i, combo in enumerate(grouped_data['Opinion_Grid'].cat.categories):
            subset = grouped_data[grouped_data['Opinion_Grid'] == combo]
            grid_size = subset['Grid Size'].iloc[0]
            
            base_index = np.arange(i, i + len(subset) * width, width)
            
            for j, (index, row) in enumerate(subset.iterrows()):
                bars = np.zeros(1)
                position = base_index[j]
                
                for k, col in enumerate(flip_columns):
                    ax.bar(
                        position,
                        row[col],
                        width=width,
                        bottom=bars[0],
                        color=colors[k],
                        label=f'{col}' if (i == 0 and j == 0) else "",
                        edgecolor='white'
                    )
                    bars += row[col]
                
                ax.text(position, 0, row['Prompt Structure'], rotation=90, ha='center', va='bottom', fontsize=9, color="white")
                
            if previous_grid_size is not None and previous_grid_size != grid_size:
                ax.axvline(x=base_index[0] - width/2, color='gray', linestyle='--')

            previous_grid_size = grid_size

        ax.set_xticks(np.arange(len(grouped_data['Opinion_Grid'].cat.categories)) + width * (len(subset) - 1) / 2)
        ax.set_xticklabels(grouped_data['Opinion_Grid'].cat.categories)
        ax.set_xlabel('Opinion Group - Grid Size')
        ax.set_ylabel('Sum of Flips')
        ax.set_title('Stacked Bar Chart of Flips by Opinion Group, Grid Size, and Prompt Structure')
        ax.legend(title='Flip Types')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(os.path.join(self._plot_path, f"{prefix}_flips.png"))

    def all_opinion_vector_viz(self, agent_vector_df: pd.DataFrame, n: int = 3):
        agent_vector_df = agent_vector_df.dropna(subset=["outcome_opinion", "my_prev_opinion", "my_opponent_prev_opinion"])
        agent_vector_df = self.uncomress_vector_index(agent_vector_df)

        print("UNIQUEQ")
        print(len(agent_vector_df["outcome_opinion"].unique()))

        self.opinion_vector_cluster_index = {}

        for prompt in agent_vector_df['prompt_structure'].unique():
            tsne_path = os.path.join(self._plot_path, f"all_opinion_embeddings_{prompt.replace('/', '_')}.png")
            tsne_kmeans_path = os.path.join(self._plot_path, f"all_opinion_embeddings_{n}_clusters_{prompt.replace('/', '_')}.png")
            tsne_bias_score = os.path.join(self._plot_path, f"all_opinion_embeddings_{n}_bias_score_{prompt.replace('/', '_')}.png")

            current_df = agent_vector_df[agent_vector_df['prompt_structure'] == prompt]

            current_df['tuple_vector'] = current_df['outcome_opinion__uncompressed'].apply(tuple)
            unique_df = current_df.drop_duplicates(subset=['outcome_opinion'])

            unique_embeddings = np.array(unique_df['tuple_vector'].apply(list).tolist())

            n = len(self._unique_bias) if self._unique_bias else n

            kmeans = KMeans(n_clusters=n, random_state=self._seed)
            unique_df['cluster'] = kmeans.fit_predict(unique_embeddings)

            for i, row in unique_df.iterrows():
                self.opinion_vector_cluster_index[f"{prompt}__{int(row['outcome_opinion'])}"] = row["cluster"] + int(self._party)

            tsne = TSNE(n_components=2, random_state=self._seed)
            unique_embeddings_2d = tsne.fit_transform(unique_embeddings)

            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=unique_embeddings_2d[:, 0], y=unique_embeddings_2d[:, 1], palette="viridis")
            plt.title(f't-SNE visualization of opinion embeddings for {prompt}')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.savefig(tsne_path)
            plt.close()

            plt.figure(figsize=(10, 8))

            sns.jointplot(
                x=unique_embeddings_2d[:, 0],
                y=unique_embeddings_2d[:, 1],
                cut = 2,
                hue=unique_df['cluster'],
                palette="viridis",
                kind='kde', fill=True,
                height=10, ratio=6,
                joint_kws = dict(alpha=0.6),
                marginal_kws=dict(fill=True)
            )
            plt.title(f't-SNE visualization for opinion embeddings by k-means labels for {prompt}')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend(title='Cluster label')
            plt.savefig(tsne_kmeans_path)
            plt.close()

            plt.figure(figsize=(10, 8))

            sns.jointplot(
                x=unique_embeddings_2d[:, 0],
                y=unique_embeddings_2d[:, 1],
                cut = 2,
                hue=unique_df['bias_score'],
                palette="viridis",
                kind='kde', fill=True,
                height=10, ratio=6,
                joint_kws = dict(alpha=0.6),
                marginal_kws=dict(fill=True)
            )
            
            plt.title(f't-SNE visualization for opinion embeddings by bias_score for {prompt}')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend(title='Bias score')
            plt.savefig(tsne_bias_score)
            plt.close()

