# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:32:27 2024

@author: 39829
"""

import argparse
import sys
import os
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection



try:
    import lanelet2
    use_lanelet2_lib = True
    #print("Successfully imported lanelet2.")
except ImportError:
    import warnings


def set_visible_area(laneletmap, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for point in laneletmap.pointLayer:
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 10, max_x + 10])
    axes.set_ylim([min_y - 10, max_y + 10])


def draw_lanelet_map(laneletmap, axes):
    assert isinstance(axes, matplotlib.axes.Axes)

    set_visible_area(laneletmap, axes)

    unknown_linestring_types = list()

    for ls in laneletmap.lineStringLayer:

        if "type" not in ls.attributes.keys():
            raise RuntimeError("ID " + str(ls.id) + ": Linestring type must be specified")
        elif ls.attributes["type"] == "curbstone":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "line_thin":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                type_dict = dict(color="darkorange", linewidth=1, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="darkorange", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "line_thick":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                type_dict = dict(color="darkorange", linewidth=2, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="darkorange", linewidth=2, zorder=10)
        elif ls.attributes["type"] == "pedestrian_marking":
            type_dict = dict(color="darkorange", linewidth=1, zorder=10, dashes=[5, 10])
        elif ls.attributes["type"] == "bike_marking":
            type_dict = dict(color="darkorange", linewidth=1, zorder=10, dashes=[5, 10])
        elif ls.attributes["type"] == "stop_line":
            type_dict = dict(color="darkorange", linewidth=3, zorder=10)
        elif ls.attributes["type"] == "virtual":
            type_dict = dict(color="darkviolet", linewidth=1, zorder=10, dashes=[2, 5])
        elif ls.attributes["type"] == "road_border":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "guard_rail":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "traffic_sign":
            continue
        elif ls.attributes["type"] == "building":
            type_dict = dict(color="pink", zorder=1, linewidth=5)
        elif ls.attributes["type"] == "spawnline":
            if ls.attributes["spawn_type"] == "start":
                type_dict = dict(color="green", zorder=11, linewidth=2)
            elif ls.attributes["spawn_type"] == "end":
                type_dict = dict(color="red", zorder=11, linewidth=2)

        else:
            if ls.attributes["type"] not in unknown_linestring_types:
                unknown_linestring_types.append(ls.attributes["type"])
            continue

        ls_points_x = [pt.x for pt in ls]
        ls_points_y = [pt.y for pt in ls]

        plt.plot(ls_points_x, ls_points_y, **type_dict)

    if len(unknown_linestring_types) != 0:
        print("Found the following unknown types, did not plot them: " + str(unknown_linestring_types))

    lanelets = []
    for ll in laneletmap.laneletLayer:
        points = [[pt.x, pt.y] for pt in ll.polygon2d()]
        polygon = Polygon(points, True)
        lanelets.append(polygon)

    #ll_patches = PatchCollection(lanelets, facecolors="lightgray", edgecolors="None", zorder=5)
    #axes.add_collection(ll_patches)
   

    if len(laneletmap.laneletLayer) == 0:
        axes.patch.set_facecolor('lightgrey')
        

    areas = []
    for area in laneletmap.areaLayer:
        if area.attributes["subtype"] == "keepout":
            points = [[pt.x, pt.y] for pt in area.outerBoundPolygon()]
            polygon = Polygon(points, True)
            areas.append(polygon)

    area_patches = PatchCollection(areas, facecolors="darkgray", edgecolors="None", zorder=5)
    axes.add_collection(area_patches)

#%%
if __name__ == "__main__":
    
    # provide data to be visualized
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat_origin", type=float,
                        help="Latitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    parser.add_argument("--lon_origin", type=float,
                        help="Longitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    args = parser.parse_args()

   
    lanelet_map_file = "DR_USA_Roundabout_FT.osm"


    # create a figure
    fig, axes = plt.subplots(1, 1)
    #fig.canvas.set_window_title("Interaction Dataset Visualization")
   
    # load and draw the lanelet2 map, either with or without the lanelet2 library
    lat_origin = args.lat_origin  # origin is necessary to correctly project the lat lon values of the map to the local
    lon_origin = args.lon_origin  # coordinates in which the tracks are provided; defaulting to (0|0) for every scenario
    print("Loading map...")

    #sets the origin of the map in terms of latitude and longitude
    #creates a UTM (Universal Transverse Mercator) projector
    projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
    laneletmap = lanelet2.io.load(lanelet_map_file, projector)
    draw_lanelet_map(laneletmap, axes)


    plt.show()

