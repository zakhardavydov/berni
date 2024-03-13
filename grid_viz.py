import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os


def generate_lattice_gif(original):

    directory = f"{original}/matrix"
    # Get list of JSON files
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    json_files.sort(key=lambda x: int(x.split('.')[0]))

    frames = []
    for json_file in json_files:
        with open(os.path.join(directory, json_file), 'r') as f:
            data = json.load(f)

        # Assuming the lattice is square
        n = int(len(data) ** 0.5)

        lattice = np.zeros((n, n))

        for key, value in data.items():
            i, j = divmod(int(key), n)
            lattice[i, j] = value['bias_score']

        # Plot lattice
        plt.figure(figsize=(6, 6))
        plt.imshow(lattice, cmap='coolwarm', interpolation='nearest')
        plt.title(f"Round {json_file.split('.')[0]}")
        plt.colorbar()

        # Save plot as image
        filename = f"round_{json_file.split('.')[0]}.png"
        plt.savefig(filename)
        plt.close()

        # Append image to frames list
        frames.append(filename)

    # Create GIF
    gif_filename = f'{original}/lattice_bias_score.gif'
    with imageio.get_writer(gif_filename, mode='I', duration=1) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)
            os.remove(frame)  # Remove the temporary image files

    print(f"GIF generated: {gif_filename}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"List of frames: {frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Visualize one simulation'
    )

    parser.add_argument("dir")

    args = parser.parse_args()

    generate_lattice_gif(args.dir)
