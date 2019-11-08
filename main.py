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

class Insect:
    def __init__(self, startpos):
        self.position = startpos

    def move(self, move_vector):
        current_x, current_y, current_z = self.position
        move_x, move_y, move_z = move_vector
        new_x = current_x + move_x
        new_y = current_y + move_y
        new_z = current_z + move_z
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

    frames = generate_grid_with_frames(100, args.dimX, args.dimY, args.dimZ)
    bug1 = Insect((10,10,10))
    bug2 = Insect((100,100,100))
    bug3 = Insect((60,40,60))
    move_bug_1 = (1,1,1)
    move_bug_2 = (-1,-1,-1)
    move_bug_3_1 = (-1,1,-1)
    move_bug_3_2 = (1,-1,1)
    for frame in range(100):
        b1_x, b1_y, b1_z = bug1.position
        b2_x, b2_y, b2_z = bug2.position
        b3_x, b3_y, b3_z = bug3.position
        frames[frame, b1_x, b1_y, b1_z] = 1
        frames[frame, b2_x, b2_y, b2_z] = 1
        frames[frame, b3_x, b3_y, b3_z] = 1
        bug1.move(move_bug_1)
        bug2.move(move_bug_2)
        if frame < 50:
            bug3.move(move_bug_3_1)
        else:
            bug3.move(move_bug_3_2)



    # save_images_folder_obj = tempfile.TemporaryDirectory()
    # save_images_folder = save_images_folder_obj.name

    # no_of_numbers = len('100')
    # for frame in range(100):
        # x_vals, y_vals, z_vals = positions_from_grid(frames[frame])
        # print(x_vals)
        # print(y_vals)
        # print(z_vals)
        # filename = {}
        # filename = f"{save_images_folder}/bugs-frame-{frame:0>{no_of_numbers}}.png"
        # save_image_from_grid(x_vals, y_vals, z_vals, filename=filename)
        # print(f"Saved frame {frame} as {filename}...")

    # # generate video
    # print("Using ffmpeg to generate avi video...")
    # commands = [
            # 'ffmpeg',
            # '-y', # Overwrite files without asking
            # '-r', # Set framerate...
            # f"25", # ...to seq_length
            # '-pattern_type', # Regextype ...
            # 'glob', # ...set to global
            # f"-i", # Pattern to use when ...
            # f"'{save_images_folder}/*.png'", # ...looking for image files
            # f"bugs-video.avi", # Where to save
            # ]
    # print(f"Running command '{' '.join(commands)}'")
    # subprocess.run(' '.join(commands), shell=True)
    # print("Dont generating video!")
    # # Clean up if we were using temporary folder for the images
    # print(f"Cleaning up temporary folder {save_images_folder}...")
    # save_images_folder_obj.cleanup()
    # print("Cleanup done.")

    # print("hello")
    # grid = generate_grid(args.dimX, args.dimY, args.dimZ)
    # print(grid.shape)
    # # Insert some random values
    # # counter = 0
    # # for x in range(128):
        # # for y in range(50, 60):
            # # for z in range(20, 60):
                # # grid[x,y,z] = 1
                # # counter += 1
    # # Simulate 3 insects 
    # grid[30,45,50] = 1
    # grid[120,10,67] = 1
    # grid[60,110,98] = 1
    # # print(f"Counter: {counter}")
    # x_vals, y_vals, z_vals = positions_from_grid(grid)
    # pprint(x_vals)
    # pprint(y_vals)
    # pprint(z_vals)
    # print_image_from_grid(x_vals, y_vals, z_vals)
    # print_image_from_grid(x_vals, y_vals, z_vals, zoom=0.7)
    # print_image_from_grid(x_vals, y_vals, z_vals, xy_angle=-10)

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
    fig = plt.figure(figsize=(side_size, side_size), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0,128)
    ax.set_ylim(0,128)
    ax.set_zlim(0,128)
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
