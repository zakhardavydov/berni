import os
import json
import argparse
import colorcet as cc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def stich_dir(dir_path: str, opinion_group_range: tuple) -> pd.DataFrame:
    batch_dfs = []

    for batch_id in os.listdir(dir_path):
        batch_path = os.path.join(dir_path, batch_id)
        if not os.path.isdir(batch_path):
            continue

        dfs = []
        for filename in os.listdir(batch_path):
            file_path = os.path.join(batch_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.csv'):
                df = pd.read_csv(file_path)
                df = df[(df['opinion_group__-1'] >= opinion_group_range[0]) & (df['opinion_group__-1'] <= opinion_group_range[1])]
                dfs.append(df)

        if dfs:
            big_df = pd.concat(dfs, ignore_index=True)

            batch_dfs.append(big_df)

    combined_df = pd.concat(batch_dfs, ignore_index=True)

    return combined_df


def plot_grid(metrics_dir: str, df: pd.DataFrame):
    palette = sns.color_palette(cc.glasbey_category10, n_colors=12)
    sns.set_theme(style="darkgrid")
    g = sns.FacetGrid(df, col="grid_size", col_wrap=4, height=3.5, palette=palette)
    g.map_dataframe(sns.lineplot, x='round', y='bias_av', hue='opinion_group__-1', estimator='mean', ci='sd', palette=palette)
    g.add_legend()
    g.set_titles(col_template='Agent Count: {col_name}')
    g.set_axis_labels('Round', 'Average Bias')

    plt.savefig(f"{metrics_dir}/opinion_dynamics.png")


def main(metrics_dir: str, opinion_group_range: tuple):
    df = stich_dir(metrics_dir, opinion_group_range)
    print(df.head(10))
    plot_grid(metrics_dir, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Metrics stitcher',
        description='Stitch run metrics and render'
    )
    parser.add_argument("dir")
    parser.add_argument("--opinion_range", nargs=2, type=float, default=[0, 1], help="Range filter for opinion_group__-1, as two integers (min max)")

    args = parser.parse_args()

    main(args.dir, args.opinion_range)
