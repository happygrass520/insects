import os
import sys
import numpy as np
import random
import argparse
import subprocess
import tempfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint

from bug_math import VelocityField, precalculate_values

class Insect:
    def __init__(self, startpos, bound):
        self.position = startpos
        self.bound = bound

    def move(self, move_vector):
        current_x, current_y, current_z = self.position
        move_x, move_y, move_z = move_vector
        new_x = current_x + move_x
        new_y = current_y + move_y
        new_z = current_z + move_z
        if new_x < 0:
            new_x = 0
        if new_x >= 128:
            new_x = 127
        if new_y < 0:
            new_y = 0
        if new_y >= 128:
            new_y = 127
        if new_z < 0:
            new_z = 0
        if new_z >= 128:
            new_z = 127
        self.position = (new_x, new_y, new_z)

def main():
    # argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('datafile', type=os.path.abspath)
    # parser.add_argument('-o', '--output', type=os.path.abspath)
    # parser.add_argument('-f', '--framerate', type=int, default=25)
    parser.add_argument('--dimX', type=int, default=128)
    parser.add_argument('--dimY', type=int, default=128)
    parser.add_argument('--dimZ', type=int, default=128)
    args = parser.parse_args()

    no_frames = 200
    frames = generate_grid_with_frames(no_frames, args.dimX, args.dimY, args.dimZ)

    # p_x, p_y, p_z = precalculate_values((128,128,128), 256)
    bound = 256
    p_x, p_y, p_z = precalculate_values((256,256,256))
    v_f = VelocityField(p_x, p_y, p_z)

    no_bugs = 10
    bugs = []
    for _ in range(no_bugs):
        x = random.randint(50, 60)
        y = random.randint(50, 60)
        z = random.randint(50, 60)
        bugs.append(Insect((x,y,z), bound))

    for frame in range(no_frames):
        for bug in bugs:
            # Print buggy
            x,y,z = bug.position
            frames[frame, x, y, z] = 1
            # Move buggy
            move_x, move_y, move_z = v_f.get_velocity(bug.position)
            print(f"Float: m_x:{move_x} m_y:{move_y} m_z:{move_z}")
            move_x = move_x * 100.0
            move_y = move_y * 100.0
            move_z = move_z * 100.0
            print(f"With gain: m_x:{move_x} m_y:{move_y} m_z:{move_z}")
            move_x, move_y, move_z = v_f.round_velocity_vector((move_x, move_y, move_z))
            print(f"Rounded: m_x:{move_x} m_y:{move_y} m_z:{move_z}")
            bug.move((move_x, move_y, move_z))
        # frame is done!

    save_video_from_grid(frames, 25, 'bugs_test_256.avi')

def save_video_from_grid(grid, framerate, video_filename):
    save_images_folder_obj = tempfile.TemporaryDirectory()
    save_images_folder = save_images_folder_obj.name

    no_of_numbers = len(str(grid.shape[0]))
    no_of_frames = grid.shape[0]
    for frame in range(no_of_frames):
        x_vals, y_vals, z_vals = positions_from_grid(grid[frame])
        filename = f"{save_images_folder}/bugs-frame-{frame:0>{no_of_numbers}}.png"
        save_image_from_grid(x_vals, y_vals, z_vals, filename=filename)
        print(f"Saved frame {frame} as {filename}...")

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
            f"'{save_images_folder}/*.png'", # ...looking for image files
            f"{video_filename}", # Where to save
            ]
    print(f"Running command '{' '.join(commands)}'")
    subprocess.run(' '.join(commands), shell=True)
    print("Dont generating video!")
    # Clean up if we were using temporary folder for the images
    print(f"Cleaning up temporary folder {save_images_folder}...")
    save_images_folder_obj.cleanup()
    print("Cleanup done.")


def save_image_from_grid(x_vals, y_vals, z_vals, elevation=30, xy_angle=-60, zoom=0, filename=None):
    dpi = 10
    side_size = 12.8
    # side_size = 25.6
    # side_size = 25.6
    fig = plt.figure(figsize=(side_size, side_size), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0,256)
    ax.set_ylim(0,256)
    ax.set_zlim(0,256)
    # ax.plot(grid[:,0], grid[:,1], grid[:,2])
    # ax.plot(x_vals, y_vals, z_vals)
    # ax.scatter(x_vals, y_vals, z_vals)
    ax.scatter(x_vals, y_vals, z_vals, c='white', depthshade=False)
    ax.view_init(elev=elevation, azim=xy_angle)
    if zoom > 0:
        ax.margins(zoom, zoom, zoom)
    else:
        ax.margins(zoom)
    ax.grid(False)
    ax.axis('off')
    ax.set_facecolor('xkcd:black')
    fig.set_facecolor('xkcd:black')
    plt.savefig(filename, dpi=dpi, edgecolor='xkcd:black', facecolor='xkcd:black')
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
