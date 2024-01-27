#!/usr/bin/env python

import sys
import argparse

# 模拟命令行参数
sys.argv = ["./main_visualize_data.py" ,
            "DR_CHN_Merging_ZS0"]

try:
    import lanelet2
    use_lanelet2_lib = True
    print("Successfully imported lanelet2.")
except ImportError:
    import warnings

    string = "Could not import lanelet2. It must be built and sourced, " + \
             "see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details."
    warnings.warn(string)
    print("Using visualization without lanelet2.")
    use_lanelet2_lib = False
    from utils import map_vis_without_lanelet

import argparse
import os
import time
import matplotlib.pyplot as plt


from utils import map_vis_lanelet2


if __name__ == "__main__":

    # provide data to be visualized
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
                                                        "files)", nargs="?")
    parser.add_argument("track_file_number", type=int, help="Number of the track file (int)", default=0, nargs="?")
    parser.add_argument("load_mode", type=str, help="Dataset to load (vehicle, pedestrian, or both)", default="both",
                        nargs="?")
    parser.add_argument("--start_timestamp", type=int, nargs="?")
    parser.add_argument("--lat_origin", type=float,
                        help="Latitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    parser.add_argument("--lon_origin", type=float,
                        help="Longitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    args = parser.parse_args()

    if args.scenario_name is None:
        raise IOError("You must specify a scenario. Type --help for help.")
    if args.load_mode != "vehicle" and args.load_mode != "pedestrian" and args.load_mode != "both":
        raise IOError("Invalid load command. Use 'vehicle', 'pedestrian', or 'both'")

    # check folders and files
    error_string = ""

    # root directory is one above main_visualize_data.py file
    # i.e. the root directory of this project
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    maps_dir = os.path.join(root_dir, "maps")

    lanelet_map_ending = ".osm"
    lanelet_map_file = os.path.join(maps_dir, args.scenario_name + lanelet_map_ending)


    # create a figure
    fig, axes = plt.subplots(1, 1)
    #fig.canvas.set_window_title("Interaction Dataset Visualization")
   
    # load and draw the lanelet2 map, either with or without the lanelet2 library
    lat_origin = args.lat_origin  # origin is necessary to correctly project the lat lon values of the map to the local
    lon_origin = args.lon_origin  # coordinates in which the tracks are provided; defaulting to (0|0) for every scenario
    print("Loading map...")
    if use_lanelet2_lib:
        #sets the origin of the map in terms of latitude and longitude
        #creates a UTM (Universal Transverse Mercator) projector
        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
        laneletmap = lanelet2.io.load(lanelet_map_file, projector)
        map_vis_lanelet2.draw_lanelet_map(laneletmap, axes)
    else:
        map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin)

    plt.show()
