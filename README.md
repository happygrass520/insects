# Insects 

Uses a Perlin noise field to simulate movement of bugs. 

Based on "Inherent noise-aware insect swarm simulation" [Wang et.al 2014](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.12277) but very simplified.

Could easily be improved by debugging the field vectors which does not seem to be working correctly.

To see options and help for running, do

```
$ python main.py -- help

usage: main.py [-h] [-o OUTPUT] [--frames FRAMES] [--framerate FRAMERATE]
               [--save_images] [--save_images_path SAVE_IMAGES_PATH]
               [--bugs BUGS] [--dimX DIMX] [--dimY DIMY] [--dimZ DIMZ]
               [--perlin_load_path PERLIN_LOAD_PATH]
               [--perlin_save_path PERLIN_SAVE_PATH] [--one_frame]
               [--angle ANGLE] [--elevation ELEVATION] [--zoom ZOOM]
               [--show_debug_grid] [--plot_vec_field] [--debug_repl]
               [--plot_alpha] [--yes_to_all] [--append_params_to_name]
               [--number_perlin_fields NUMBER_PERLIN_FIELDS]
               [--switch_fields_every_frame SWITCH_FIELDS_EVERY_FRAME]

Generate some insects flying infront of the camera

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        filename (without extenstion) for the finished .avi
                        file
  --frames FRAMES       How many frames to generate
  --framerate FRAMERATE
                        What framerate to generate the video with
  --save_images         If true, images are saved to a permanent folder,
                        otherwise tempfolder is used
  --save_images_path SAVE_IMAGES_PATH
                        Where to save the images
  --bugs BUGS           How many bugs to render
  --dimX DIMX           Define dimension of the X axis
  --dimY DIMY           Define dimension of the Y axis
  --dimZ DIMZ           Define dimension of the Z axis
  --perlin_load_path PERLIN_LOAD_PATH
                        Folder to look for saved perlin files
  --perlin_save_path PERLIN_SAVE_PATH
                        Folder to save perlin files in
  --one_frame           Shows one frame and then quits - for debugging
  --angle ANGLE         Passed to pyplot
  --elevation ELEVATION
                        Passed to pyplot
  --zoom ZOOM           Pyplot zoom - define start-end on axis as ints (ex
                        --zoom 130-170). Check what you want with
                        --show_debug_grid and --one_frame
  --show_debug_grid     Show the axis and plot dimensions
  --plot_vec_field      Plots the vector field of the last Perlin field
                        constructed, then quits - for debugging
  --debug_repl          Loads a repl for the last Perlin field constructed,
                        then quits - for debugging
  --plot_alpha          Plot the alpha function from vector field, then quits
  --yes_to_all          Dont stop to ask questions, just go. !! Will overwrite
                        things !!
  --append_params_to_name
                        Put a bunch of parameters in the name of the file, to
                        keep track of how it was generated
  --number_perlin_fields NUMBER_PERLIN_FIELDS
                        How many perlin fields to use
  --switch_fields_every_frame SWITCH_FIELDS_EVERY_FRAME
                        Every x frame switch perlin field to the next one
```

## Requirements
* Python 3.6+
* Matplotlib
* Numpy


