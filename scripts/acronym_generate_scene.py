#!/usr/bin/env python3
"""
The MIT License (MIT)

Copyright (c) 2020 NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import sys
import json
import h5py
import trimesh
from pathlib import Path
import argparse
import numpy as np
import trimesh.path
from shapely.geometry import Point

from acronym_tools import Scene, load_mesh, load_grasps, create_gripper_marker


def make_parser():
    parser = argparse.ArgumentParser(
        description="Generate a random scene arrangement and filtering grasps that are in collision.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--objects", nargs="+", help="HDF5 or JSON Object file(s).")
    parser.add_argument(
        "--support",
        required=True,
        type=str,
        help="HDF5 or JSON File for support object.",
    )
    parser.add_argument(
        "--support_scale", default=0.025, help="Scale factor of support mesh."
    )
    parser.add_argument(
        "--show_grasps",
        action="store_true",
        help="Show all grasps that are not in collision.",
    )
    parser.add_argument(
        "--mesh_root", default=".", help="Directory used for loading meshes."
    )
    parser.add_argument(
        "--num_grasps_per_object",
        default=20,
        help="Maximum number of grasps to show per object.",
    )
    return parser


def main(argv=sys.argv[1:]):
    parser = make_parser()
    args = parser.parse_args(argv)

    # load object meshes
    object_meshes = [load_mesh(o, mesh_root_dir=args.mesh_root) for o in args.objects]
    support_mesh = load_mesh(
        args.support, mesh_root_dir=args.mesh_root, scale=args.support_scale
    )
    scene = Scene.random_arrangement(object_meshes, support_mesh)

    # show the random scene in 3D viewer
    scene.colorize().as_trimesh_scene().show()

    if args.show_grasps:
        # load gripper mesh for collision check
        gripper_mesh = trimesh.load(
            Path(__file__).parent.parent / "data/franka_gripper_collision_mesh.stl"
        )
        gripper_markers = []
        for i, fname in enumerate(args.objects):
            T, success = load_grasps(fname)
            obj_pose = scene._poses["obj{}".format(i)]

            # check collisions
            collision_free = np.array(
                [
                    i
                    for i, t in enumerate(T[success == 1][: args.num_grasps_per_object])
                    if not scene.in_collision_with(
                        gripper_mesh, transform=np.dot(obj_pose, t)
                    )
                ]
            )

            if len(collision_free) == 0:
                continue

            # add a gripper marker for every collision free grasp
            gripper_markers.extend(
                [
                    create_gripper_marker(color=[0, 255, 0]).apply_transform(
                        np.dot(obj_pose, t)
                    )
                    for t in T[success == 1][collision_free]
                ]
            )

        # show scene together with successful and collision-free grasps of all objects
        trimesh.scene.scene.append_scenes(
            [scene.colorize().as_trimesh_scene(), trimesh.Scene(gripper_markers)]
        ).show()


if __name__ == "__main__":
    main()
