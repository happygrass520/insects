import os
import sys
import numpy as np
import random
import argparse
import pickle
import subprocess
import tempfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint

from bug_math import VelocityField, precalculate_values

class Insect:
    def __init__(self, startpos, bound_x, bound_y, bound_z):
        self.position = startpos
        self.bound_x = bound_x
        self.bound_y = bound_y
        self.bound_z= bound_z

    def move(self, move_vector):
        current_x, current_y, current_z = self.position
        move_x, move_y, move_z = move_vector
        new_x = current_x + move_x
        new_y = current_y + move_y
        new_z = current_z + move_z
        if new_x < 0 or new_x >= self.bound_x:
            new_x = current_x - move_x
        if new_y < 0 or new_y >= self.bound_y:
            new_y = current_y - move_y
        if new_z < 0 or new_z >= self.bound_z:
            new_z = current_z - move_z
        self.position = (new_x, new_y, new_z)

def main():
    # argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('datafile', type=os.path.abspath)
    parser.add_argument('-o', '--output', type=os.path.abspath)
    # parser.add_argument('-f', '--framerate', type=int, default=25)
    parser.add_argument('--frames', type=int, default=150)
    parser.add_argument('--dimX', type=int, default=128)
    parser.add_argument('--dimY', type=int, default=128)
    parser.add_argument('--dimZ', type=int, default=128)
    parser.add_argument('--perlin_load_path', type=os.path.abspath)
    parser.add_argument('--perlin_save_path', type=os.path.abspath)
    args = parser.parse_args()

    no_frames = args.frames
    bound_x = args.dimX
    bound_y = args.dimY
    bound_z = args.dimZ

    if args.perlin_load_path is None:
        print("No perlin path, calculating new ones")
        p_x, p_y, p_z = precalculate_values((bound_x,bound_y,bound_z))
    else:
        print(f"Loading perlin from {args.perlin_load_path}...")
        p_x = load_perlin_noise(args.perlin_load_path, 'p_x', bound_x)
        print("Loaded p_x")
        p_y = load_perlin_noise(args.perlin_load_path, 'p_y', bound_y)
        print("Loaded p_y")
        p_z = load_perlin_noise(args.perlin_load_path, 'p_z', bound_z)
        print("Loaded p_z")

    if args.perlin_save_path is not None:
        print(f"Saving perlin to {args.perlin_save_path}...")
        save_perlin_noise(args.perlin_save_path, 'p_x', p_x, bound_x)
        print("Saved p_x...")
        save_perlin_noise(args.perlin_save_path, 'p_y', p_y, bound_y)
        print("Saved p_y...")
        save_perlin_noise(args.perlin_save_path, 'p_z', p_z, bound_z)
        print("Saved p_z! Done!")

    v_f = VelocityField(p_x, p_y, p_z)

    no_bugs = 10
    bugs = []
    for _ in range(no_bugs):
        x = random.randint(0,bound_x)
        y = random.randint(0,bound_y)
        z = random.randint(0,bound_z)
        bugs.append(Insect((x,y,z), bound_x, bound_y, bound_z))

    # save_images_folder_obj = tempfile.TemporaryDirectory()
    # save_images_folder = save_images_folder_obj.name
    save_images_folder = 'images'
    if not os.path.isdir(save_images_folder):
        os.mkdir(save_images_folder)

    noise_gain = 200.0
    no_of_numbers = len(str(no_frames))
    frame_counter = 0
    for frame in range(no_frames):
        frame = np.zeros((bound_x,bound_y,bound_z))
        print("Moving bugs...", end='')
        for bug in bugs:
            # Print buggy
            x,y,z = bug.position
            frame[x, y, z] = 1
            # Move buggy
            move_x, move_y, move_z = v_f.get_velocity(bug.position)
            move_x = move_x * noise_gain
            move_y = move_y * noise_gain
            move_z = move_z * noise_gain
            move_x, move_y, move_z = v_f.round_velocity_vector((move_x, move_y, move_z))
            bug.move((move_x, move_y, move_z))
        print("Generating frame...", end='')
        x_vals, y_vals, z_vals = positions_from_grid(frame)
        filename = f"{save_images_folder}/bugs-frame-{frame_counter:0>{no_of_numbers}}.png"
        save_image_from_grid(x_vals, y_vals, z_vals, filename=filename)
        print(f"Saved frame {frame_counter} as {filename}!\r", end='')
        # frame is done!
        frame_counter += 1

    save_video_from_grid(save_images_folder, 25, args.output)

    print(f"Cleaning up temporary folder {save_images_folder}...")
    # save_images_folder_obj.cleanup()
    print("Cleanup done.")

def save_video_from_grid(frames_folder, framerate, video_filename):
    # generate video
    print("Using ffmpeg to generate avi video...")
    commands = [
            'ffmpeg',
            '-y', # Overwrite files without asking
            '-r', # Set framerate...
            f"{framerate}", # ...to seq_length
            '-pattern_type', # Regextype ...
            'glob', # ...set to global
            f"-i", # Pattern to use when ...
            f"'{frames_folder}/*.png'", # ...looking for image files
            f"{video_filename}", # Where to save
            ]
    print(f"Running command '{' '.join(commands)}'")
    subprocess.run(' '.join(commands), shell=True)
    print("Dont generating video!")

def save_perlin_noise(folder, filename, p, dimension):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    with open(f'{folder}/{filename}_{dimension}.perlin', 'wb') as f:
        pickle.dump(p, f)

def load_perlin_noise(folder, filename, dimension):
    loaded_p = None
    with open(f'{folder}/{filename}_{dimension}.perlin', 'rb') as f:
        loaded_p = pickle.load(f)
    return loaded_p

# def save_image_from_grid(x_vals, y_vals, z_vals, elevation=30, xy_angle=-60, zoom=0, filename=None):
def save_image_from_grid(x_vals, y_vals, z_vals, elevation=-19, xy_angle=67, zoom=0, filename=None):
    dpi = 10
    side_size = 12.8
    # side_size = 25.6
    # side_size = 25.6
    # fig = plt.figure(figsize=(side_size, side_size), dpi=dpi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0,256)
    ax.set_ylim(0,256)
    ax.set_zlim(0,256)
    # ax.plot(grid[:,0], grid[:,1], grid[:,2])
    # ax.plot(x_vals, y_vals, z_vals)
    # ax.scatter(x_vals, y_vals, z_vals)
    # ax.scatter(x_vals, y_vals, z_vals, c='white', depthshade=False)
    ax.scatter(x_vals, y_vals, z_vals, c='white', depthshade=True)
    ax.view_init(elev=elevation, azim=xy_angle)
    if zoom > 0:
        ax.margins(zoom, zoom, zoom)
    # else:
        # ax.margins(zoom)
    ax.grid(False)
    ax.axis('off')
    ax.set_facecolor('xkcd:black')
    fig.set_facecolor('xkcd:black')
    # plt.savefig(filename, dpi=dpi, edgecolor='xkcd:black', facecolor='xkcd:black')
    plt.savefig(filename, edgecolor='xkcd:black', facecolor='xkcd:black')
    # plt.savefig(filename)
    # plt.savefig(filename, dpi=10)
    # plt.show()
    plt.close(fig)

def positions_from_grid(grid):
    """
    Takes the grid and returns coordinates, starting from (0,0,0)->(dimX,dimY,dimZ)
    """
    return np.nonzero(grid)

def generate_grid(dimX, dimY, dimZ):
    return np.zeros((dimX, dimY, dimZ))

def generate_grid_with_frames(frames, dimX, dimY, dimZ):
    return np.zeros((frames, dimX, dimY, dimZ))

if __name__=='__main__':
    main()
