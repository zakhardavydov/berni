import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import imageio

from PIL import Image, ImageSequence


matplotlib.use("Agg")


def generate_lattice_gif(directory_path, output_gif_path, override: bool = True):

    if not override and os.path.exists(output_gif_path):
        return output_gif_path
    
       # Get list of runs (subdirectories)
    runs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

    if not runs:
        return None
    
    # Sort runs to ensure consistent order
    
    run_meta = {}

    for run in runs:
        df_path = os.path.join(directory_path, f"{run}.csv")
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            run_meta[run] = {"opinion_group__-1": df["opinion_group__-1"][0]}
        else:
            runs = [r for r in runs if r != run]

    runs = sorted(runs, key=lambda run: run_meta[run]["opinion_group__-1"])

    if not runs:
        return None
    
    # Get the number of rounds from the first run
    rounds = len([f for f in os.listdir(os.path.join(directory_path, runs[0], 'matrix')) if f.endswith('.json')])
    
    images = []
    for frame in range(1, rounds):
        grid_size = math.ceil(np.sqrt(len(runs)))
        if grid_size == 1:
            grid_size = 2
        plt.figure(figsize=(20, 20))
        fig, ax = plt.subplots(grid_size, grid_size)
        fig.tight_layout()
        for i, run in enumerate(runs):
            run_path = os.path.join(directory_path, run, 'matrix', f'{frame}.json')
            with open(run_path) as f:
                data = json.load(f)
                lattice_size = int(np.sqrt(len(data)))
                lattice = np.zeros((lattice_size, lattice_size))
                for idx, agent_data in data.items():
                    idx = int(idx)  # Convert index to integer
                    row = idx // lattice_size
                    col = idx % lattice_size
                    lattice[row, col] = agent_data['bias_score']
                row_index = i // grid_size
                col_index = i % grid_size
                im = ax[row_index, col_index].imshow(lattice, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
                ax[row_index, col_index].set_title(f'Round {frame}, Op {run_meta[run]["opinion_group__-1"]}', fontsize=8)
                plt.colorbar(im)
        # Save current frame to image
        image_path = f"temp_{frame}.png"
        plt.savefig(image_path)
        images.append(imageio.imread(image_path))
        os.remove(image_path)
        plt.close(fig)
        
    # Save images to GIF
    imageio.mimsave(output_gif_path, images, duration=1)
    return output_gif_path


def get_max_frames(gifs):
    """Return the maximum number of frames among the input GIFs."""
    return max(len(list(ImageSequence.Iterator(gif))) for gif in gifs)


def create_gif_grid(gif_paths, output_path):
    # Load the GIFs
    gifs = [Image.open(path) for path in gif_paths]
    grid_size = int(np.ceil(np.sqrt(len(gifs))))  # Determine grid size

    # Assume all GIFs are the same size for simplicity
    gif_width, gif_height = gifs[0].size
    max_frames = get_max_frames(gifs)

    # Frame list to hold each combined frame
    frames = []

    # Create each frame
    for frame_index in range(max_frames):
        # Create a new blank canvas for this frame
        canvas = Image.new('RGBA', (gif_width * grid_size, gif_height * grid_size))

        for index, gif in enumerate(gifs):
            # Calculate grid position
            x = (index % grid_size) * gif_width
            y = (index // grid_size) * gif_height

            try:
                gif.seek(frame_index)  # Move to the frame_index frame
                frame = gif.copy()  # Copy the current frame
            except EOFError:
                # If this GIF doesn't have a frame at frame_index, reuse its last frame
                pass

            canvas.paste(frame, (x, y))

        frames.append(canvas)

    # Save the frames as a new animated GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)



def exp_grid(dir: str):
    exps = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    output = [generate_lattice_gif(exp, f"{exp}/grid.gif", override=False) for exp in exps]
    output = [o for o in output if o]
    create_gif_grid(output, f"{dir}/grid.gif")    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Visualize a batch of simulatsions"
    )

    parser.add_argument("dir")
    parser.add_argument('--batch', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    if args.batch:
        generate_lattice_gif(args.dir, f"{args.dir}/grid.gif")
    else:
        exp_grid(args.dir)
