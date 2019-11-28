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

from bug_math import VelocityField, precalculate_values, generate_perlin_noise_3d

class Insect:
    def __init__(self, startpos, bound_x, bound_y, bound_z, name):
        self.name = name
        x,y,z = startpos
        x = float(x)
        y = float(y)
        z = float(z)
        self.position = (x,y,z)
        self.bound_x = float(bound_x)
        self.bound_y = float(bound_y)
        self.bound_z = float(bound_z)

    def move(self, move_vector):
        current_x, current_y, current_z = self.position
        move_x, move_y, move_z = move_vector
        new_x = current_x + move_x
        new_y = current_y + move_y
        new_z = current_z + move_z
        # Testing with removing this check
        if new_x < 0.0 or new_x > self.bound_x - 1.0:
            new_x = current_x - move_x
        if new_y < 0.0 or new_y > self.bound_y - 1.0:
            new_y = current_y - move_y
        if new_z < 0.0 or new_z > self.bound_z - 1.0:
            new_z = current_z - move_z
        self.position = (new_x, new_y, new_z)

    def get_rounded_position(self):
        x,y,z = self.position
        return (int(x), int(y), int(z))

    def __str__(self):
        x,y,z = self.position
        return f"Bug {self.name}: {x:.2f}, {y:.2f}, {z:.2f}"

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
    parser.add_argument('--one_frame', action='store_true', help='Shows one frame and then quits')
    parser.add_argument('--angle', type=int, default=40)
    parser.add_argument('--elevation', type=int, default=6)
    parser.add_argument('--zoom', type=float, default=0.0)
    parser.add_argument('--show_debug_grid', action='store_true')
    parser.add_argument('--plot_vec_field', action='store_true')
    parser.add_argument('--yes_to_all', action='store_true')
    args = parser.parse_args()

    no_frames = args.frames
    bound_x = args.dimX
    bound_y = args.dimY
    bound_z = args.dimZ

    # v_f = VelocityField(None, None, None, bound_x, bound_y, bound_z)
    # Load or generate perlin noise
    p_x, p_y, p_z = perlin_values((bound_x, bound_y, bound_z), args.perlin_load_path, args.perlin_save_path, args.yes_to_all)

    v_f = VelocityField(p_x, p_y, p_z, bound_x, bound_y, bound_z)

    if args.plot_vec_field:
        # v_f.plot_vec_field(step_size=16)
        v_f.plot_alpha_ramp()
        v_f.plot_vec_field(step_size=2)

    # v_f.plot_vec_field(step_size=32)
    # v_f.plot_vec_field(step_size=1)

    no_bugs = 10
    bugs = []
    for i in range(no_bugs):
        x = random.randint(0,bound_x)
        y = random.randint(0,bound_y)
        z = random.randint(0,bound_z)
        bugs.append(Insect((float(x),float(y),float(z)), float(bound_x), float(bound_y), float(bound_z), f"{i}"))

    save_images_folder_obj = tempfile.TemporaryDirectory()
    save_images_folder = save_images_folder_obj.name
    # save_images_folder = 'images'
    if not os.path.isdir(save_images_folder):
        os.mkdir(save_images_folder)

    # noise_gain = 200.0
    no_of_numbers = len(str(no_frames))
    frame_counter = 0
    for frame in range(no_frames):
        frame = np.zeros((bound_x,bound_y,bound_z))
        print("Moving bugs...", end='')
        for bug in bugs:
            # Print buggy
            x,y,z = bug.get_rounded_position()
            frame[x, y, z] = 1
            # Move buggy
            move_x, move_y, move_z = v_f.get_velocity(bug.get_rounded_position())
            # print(f"move:({move_x}, {move_y}, {move_z})")
            # print(bug)
            # move_x = move_x * noise_gain
            # move_y = move_y * noise_gain
            # move_z = move_z * noise_gain
            # move_x, move_y, move_z = v_f.round_velocity_vector((move_x, move_y, move_z))
            # print(f"move rounding:({move_x}, {move_y}, {move_z})")
            bug.move((move_x, move_y, move_z))
        print("Generating frame...", end='')
        x_vals, y_vals, z_vals = positions_from_grid(frame)
        filename = f"{save_images_folder}/bugs-frame-{frame_counter:0>{no_of_numbers}}.png"
        if args.one_frame:
            show_image_from_grid(
                    x_vals,
                    y_vals,
                    z_vals,
                    filename=filename,
                    elevation=args.elevation,
                    xy_angle=args.angle,
                    zoom=args.zoom,
                    show_debug_grid=args.show_debug_grid,
                    )
            quit()
        save_image_from_grid(
                x_vals,
                y_vals,
                z_vals,
                filename=filename,
                elevation=args.elevation,
                xy_angle=args.angle,
                zoom=args.zoom,
                show_debug_grid=args.show_debug_grid,
                )
        print(f"Saved frame {frame_counter} as {filename}!\r", end='')
        # frame is done!
        frame_counter += 1

    save_video_from_grid(save_images_folder, 25, args.output)

    print(f"Cleaning up temporary folder {save_images_folder}...")
    save_images_folder_obj.cleanup()
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
    try:
        with open(f'{folder}/{filename}_{dimension}.perlin', 'rb') as f:
            loaded_p = pickle.load(f)
        return loaded_p
    except OSError:
        # We couldnt find it
        return None

def perlin_values(bounds, load_path, save_path, yes_to_all):
    # Define defaults
    # Resolution for perlin noise.. maybe
    res = (4,4,4)
    b_x, b_y, b_z = bounds
    p_x = None
    p_y = None
    p_z = None
    # Check if files has been loaded
    loaded_p_x = False
    loaded_p_y = False
    loaded_p_z = False
    # Lets see if we can load any!
    if load_path is not None:
        print("Trying to load p_x...")
        l_p_x = load_perlin_noise(load_path, 'p_x', b_x)
        if l_p_x is None:
            print(f"Could not load p_x for {b_x}, creating...")
            l_p_x = generate_perlin_noise_3d(bounds,res)
            print("Done")
        else:
            print("Successful loading of p_x!")
            loaded_p_x = True
        p_x = l_p_x

        print("Trying to load p_y...")
        l_p_y = load_perlin_noise(load_path, 'p_y', b_y)
        if l_p_y is None:
            print(f"Could not load p_y for {b_y}, creating...")
            l_p_y = generate_perlin_noise_3d(bounds,res)
            print("Done")
        else:
            print("Successful loading of p_y!")
            loaded_p_y = True
        p_y = l_p_y

        print("Trying to load p_z...")
        l_p_z = load_perlin_noise(load_path, 'p_z', b_z)
        if l_p_z is None:
            print(f"Could not load p_z for {b_z},creating...")
            l_p_z = generate_perlin_noise_3d(bounds,res)
            print("Done")
        else:
            print("Successful loading of p_z!")
            loaded_p_z = True
        p_z = l_p_z
    # now we know that p_[x,y,z] are filled with values
    # Lets see if we wanna save it
    if save_path is not None:
        if loaded_p_x:
            print("p_x was loaded, skipping save...")
        else:
            print("Saving p_x...")
            save_perlin_noise(save_path, 'p_x', p_x, b_x)
            print("Saved!")
        if loaded_p_y:
            print("p_y was loaded, skipping save...")
        else:
            print("Saving p_y...")
            save_perlin_noise(save_path, 'p_y', p_y, b_y)
            print("Saved!")
        if loaded_p_z:
            print("p_z was loaded, skipping save...")
        else:
            print("Saving p_z...")
            save_perlin_noise(save_path, 'p_z', p_z, b_z)
            print("Saved!")

    return p_x, p_y, p_z




def generate_image(x_vals, y_vals, z_vals, elevation, xy_angle, zoom, show_debug_grid):
    dpi = 10
    side_size = 12.8
    # side_size = 25.6
    # fig = plt.figure(figsize=(side_size, side_size), dpi=dpi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0,128)
    ax.set_ylim(0,128)
    ax.set_zlim(0,128)
    if show_debug_grid:
        ax.scatter(x_vals, y_vals, z_vals, depthshade=True)
    else:
        ax.scatter(x_vals, y_vals, z_vals, c='white', depthshade=True)
    ax.view_init(elev=elevation, azim=xy_angle)
    if zoom > 0:
        ax.margins(zoom, zoom, zoom)
    if not show_debug_grid:
        ax.grid(False)
        ax.axis('off')
        ax.set_facecolor('xkcd:black')
        fig.set_facecolor('xkcd:black')
    return ax, fig

# def save_image_from_grid(x_vals, y_vals, z_vals, elevation=30, xy_angle=-60, zoom=0, filename=None):
def save_image_from_grid(x_vals, y_vals, z_vals, elevation=-19, xy_angle=67, zoom=0, filename=None, show_debug_grid=False):
    ax, fig = generate_image(x_vals, y_vals, z_vals, elevation, xy_angle, zoom, show_debug_grid)
    # plt.savefig(filename, dpi=dpi, edgecolor='xkcd:black', facecolor='xkcd:black')
    # plt.savefig(filename, edgecolor='xkcd:black', facecolor='xkcd:black')
    plt.savefig(filename)
    # plt.savefig(filename, dpi=10)
    # plt.show()
    plt.close(fig)

def show_image_from_grid(x_vals, y_vals, z_vals, elevation=-19, xy_angle=67, zoom=0, filename=None, show_debug_grid=False):
    ax, fig = generate_image(x_vals, y_vals, z_vals, elevation, xy_angle, zoom, show_debug_grid)
    # plt.savefig(filename, dpi=dpi, edgecolor='xkcd:black', facecolor='xkcd:black')
    # plt.savefig(filename, edgecolor='xkcd:black', facecolor='xkcd:black')
    # plt.savefig(filename)
    # plt.savefig(filename, dpi=10)
    # plt.show(dpi=10)
    plt.show()
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
